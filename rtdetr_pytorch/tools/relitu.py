import cv2
import numpy as np
import os
import sys
import torch
from torch.cuda.amp import autocast
from torchvision.transforms import Compose, ToTensor, Resize
import torch.nn.functional as F
# 添加当前工作目录到 Python 模块搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 添加 src 模块所在目录到 Python 模块搜索路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# 定义设备变量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用相对导入
from src.nn.backbone import PResNet
from src.zoo.rtdetr import HybridEncoder, RTDETRTransformer, RTDETR, RTDETRPostProcessor
from PIL import Image
from torchvision import transforms


backbone = PResNet(depth=18, freeze_norm=False, pretrained=False, return_idx=[1, 2, 3])
encoder = HybridEncoder(in_channels=[128, 256, 512], enc_act='gelu', expansion=0.5, eval_spatial_size=[640, 640])
decoder = RTDETRTransformer(
    num_classes=6,
    hidden_dim=256,
    num_decoder_layers=3,
    num_denoising=100,
    feat_channels=[256, 256, 256],
    feat_strides=[8, 16, 32],
    eval_spatial_size=[640, 640]
)
model = RTDETR(backbone=backbone, encoder=encoder, decoder=decoder, multi_scale=[640]).to(device)
postprocessor = RTDETRPostProcessor(num_classes=6, remap_mscoco_category=True)

# 加载权重
weights_path = "outputs/deeppcb/checkpoint0071.pth"
weights = torch.load(weights_path)
model.load_state_dict(weights['model'])
model.eval()

# # 打印模型结构
# print("Model structure:")
# print(model)

# 选择目标层（选择输入投影层的最后一个卷积层）
target_layer = model.decoder.input_proj[-1].conv  # 选择具有空间维度的卷积层
print(f"Selected target layer: {target_layer}")


# ===== 2. Grad-CAM 实现（适配 Transformer 结构） =====
class GradCAM:
    def __init__(self, model, target_layer):
        """初始化并注册前向/反向钩子"""
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # 注册钩子
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        """保存前向传播的激活值"""
        self.activations = output
    
    def save_gradients(self, module, grad_input, grad_output):
        """保存反向传播的梯度"""
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, target_class):
        # 清除之前的梯度
        self.model.zero_grad()
        # 前向传播
        output = self.model(input_tensor)
        logits = output['pred_logits']
        
        # Grad-CAM++ 实现
        if hasattr(self.model, 'decoder'):
            # 修改：使用正确的属性名
            try:
                # 尝试访问解码器的最后一层注意力权重
                attention_weights = self.model.decoder.layers[-1].cross_attn.attn
            except AttributeError:
                # 若属性名不同，可尝试其他可能的属性名
                print("Warning: Unable to access attention weights. Using fallback method.")
                if self.gradients is not None:
                    weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
                    cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
                else:
                    cam = torch.zeros_like(self.activations)
                # 增强的归一化处理
                cam = F.relu(cam)
                cam = F.interpolate(cam.unsqueeze(0), input_tensor.shape[2:], mode='bilinear', align_corners=False)
                cam = cam.squeeze().cpu().numpy()
                cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
                return cam
            
            # 使用类别置信度加权注意力图
            class_attention = attention_weights[:, :, target_class].mean(0)
            
            # 计算高阶梯度
            loss = (logits[:, :, target_class] * class_attention.unsqueeze(0)).sum()
            loss.backward(retain_graph=True)
            
            # 改进的alpha计算（考虑二阶导数）
            second_order_grads = torch.autograd.grad(loss, self.activations, create_graph=True)[0]
            alpha_num = second_order_grads.pow(2)
            alpha_denom = 2 * alpha_num + (second_order_grads * self.activations).pow(3).sum(dim=(2,3), keepdim=True)
            alpha_denom = alpha_denom.clamp(min=1e-8)
            alpha = alpha_num / alpha_denom
            
            # ReLU激活和权重计算
            weights = alpha * torch.relu(self.gradients)
            cam = torch.sum(weights * self.activations, dim=(0,1))
        else:
            # 传统Grad-CAM回退
            if self.gradients is not None:
                weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            else:
                cam = torch.zeros_like(self.activations)

        # 增强的归一化处理
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0), input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        return cam


# 初始化 Grad-CAM
cam_extractor = GradCAM(model, target_layer)


# ===== 3. 批量处理函数 =====
def generate_heatmaps(image_dir, output_dir, target_class=0, u_shape_heatmap=False):
    os.makedirs(output_dir, exist_ok=True)
    transform = Compose([ToTensor(), Resize((640, 640), antialias=True)])

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(image_dir, filename)

        # 加载图像（PIL 和 OpenCV 同步加载）
        pil_image = Image.open(image_path).convert("RGB")
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            print(f"Warning: Failed to load {image_path}")
            continue

        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        # 生成热力图
        cam = cam_extractor.generate_cam(input_tensor, target_class=target_class)
        heatmap = cv2.resize(cam, (pil_image.width, pil_image.height))
        heatmap = np.uint8(255 * heatmap)

        # 如果启用U形热力图，调整热力图权重分布
        if u_shape_heatmap:
            # 改进的U形权重计算（动态高斯核）
            h, w = heatmap.shape
            y_center, x_center = h//2, w//2
            y, x = np.ogrid[:h, :w]
            
            # 动态调整参数（基于图像尺寸）
            sigma_y = max(h/6, 1)
            sigma_x = max(w/6, 1)
            
            # 高斯衰减函数
            u_weight = np.exp(-((y - y_center)**2/(2*sigma_y**2) + 
                              (x - x_center)**2/(2*sigma_x**2)))
            
            # 应用注意力引导的增强
            if 'attention_weights' in locals():
                # 将注意力权重上采样到图像尺寸
                att_map = F.interpolate(attention_weights.unsqueeze(0), 
                                      size=(h,w), 
                                      mode='bilinear', 
                                      align_corners=False)
                # 结合注意力权重
                u_weight = u_weight * (1 + att_map.cpu().numpy()[0])
            
            # 归一化处理
            u_weight = (u_weight - u_weight.min()) / (u_weight.max() - u_weight.min() + 1e-8)
            heatmap = np.uint8(heatmap * u_weight)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 确保 OpenCV 图像和热力图尺寸一致
        h, w, _ = cv_image.shape
        heatmap = cv2.resize(heatmap, (w, h))

        # 叠加图像（确保通道顺序正确）
        superimposed_img = cv2.addWeighted(cv_image, 0.6, heatmap, 0.4, 0)

        save_path = os.path.join(output_dir, f"cam_{filename}")
        cv2.imwrite(save_path, superimposed_img)
        print(f"Saved: {save_path}")

# ===== 4. 执行处理 =====
if __name__ == "__main__":
    image_dir = "outputs/deeppcb-infer/relitu"
    output_dir = "outputs/deeppcb-infer/result"
    target_class = 0  # 确保在 0 <= target_class < 6 范围内
    u_shape_heatmap = True  # 启用U形热力图

    generate_heatmaps(image_dir, output_dir, target_class=target_class, u_shape_heatmap=u_shape_heatmap)
