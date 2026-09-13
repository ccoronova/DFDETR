import cv2
import numpy as np
import os
import sys
import torch
from torch.cuda.amp import autocast
from torchvision.transforms import Compose, ToTensor, Resize
import torch.nn.functional as F
# Add the current working directory to the Python module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add the directory containing the src module to the Python module search path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Define the device variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use relative imports
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

# Load weights
weights_path = "outputs/deeppcb/checkpoint0071.pth"
weights = torch.load(weights_path)
model.load_state_dict(weights['model'])
model.eval()

# # Print model structure
# print("Model structure:")
# print(model)

# Select the target layer (select the last convolutional layer of the input projection layer)
target_layer = model.decoder.input_proj[-1].conv  # Select the convolutional layer with spatial dimensions
print(f"Selected target layer: {target_layer}")


# ===== 2. Grad-CAM implementation (adapted to Transformer architecture) =====
class GradCAM:
    def __init__(self, model, target_layer):
        """Initialize and register forward/backward hooks"""
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        """Save the activations from forward propagation"""
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        """Save the gradients from backward propagation"""
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class):
        # Clear previous gradients
        self.model.zero_grad()
        # Forward propagation
        output = self.model(input_tensor)
        logits = output['pred_logits']
        
        # Grad-CAM++ implementation
        if hasattr(self.model, 'decoder'):
            # Modification: use the correct attribute name
            try:
                # Try to access the last decoder layer's attention weights
                attention_weights = self.model.decoder.layers[-1].cross_attn.attn
            except AttributeError:
                # If the attribute name differs, try other possible attribute names
                print("Warning: Unable to access attention weights. Using fallback method.")
                if self.gradients is not None:
                    weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
                    cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
                else:
                    cam = torch.zeros_like(self.activations)
                # Enhanced normalization
                cam = F.relu(cam)
                cam = F.interpolate(cam.unsqueeze(0), input_tensor.shape[2:], mode='bilinear', align_corners=False)
                cam = cam.squeeze().cpu().numpy()
                cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
                return cam

            # Use class-confidence weighted attention map
            class_attention = attention_weights[:, :, target_class].mean(0)
            
            # Compute higher-order gradients
            loss = (logits[:, :, target_class] * class_attention.unsqueeze(0)).sum()
            loss.backward(retain_graph=True)

            # Improved alpha computation (considering second-order derivatives)
            second_order_grads = torch.autograd.grad(loss, self.activations, create_graph=True)[0]
            alpha_num = second_order_grads.pow(2)
            alpha_denom = 2 * alpha_num + (second_order_grads * self.activations).pow(3).sum(dim=(2,3), keepdim=True)
            alpha_denom = alpha_denom.clamp(min=1e-8)
            alpha = alpha_num / alpha_denom

            # ReLU activation and weight computation
            weights = alpha * torch.relu(self.gradients)
            cam = torch.sum(weights * self.activations, dim=(0,1))
        else:
            # Traditional Grad-CAM fallback
            if self.gradients is not None:
                weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            else:
                cam = torch.zeros_like(self.activations)

        # Enhanced normalization
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0), input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        return cam


# Initialize Grad-CAM
cam_extractor = GradCAM(model, target_layer)


# ===== 3. Batch processing function =====
def generate_heatmaps(image_dir, output_dir, target_class=0, u_shape_heatmap=False):
    os.makedirs(output_dir, exist_ok=True)
    transform = Compose([ToTensor(), Resize((640, 640), antialias=True)])

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(image_dir, filename)

        # Load the image (PIL and OpenCV loaded simultaneously)
        pil_image = Image.open(image_path).convert("RGB")
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            print(f"Warning: Failed to load {image_path}")
            continue

        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        # Generate the heatmap
        cam = cam_extractor.generate_cam(input_tensor, target_class=target_class)
        heatmap = cv2.resize(cam, (pil_image.width, pil_image.height))
        heatmap = np.uint8(255 * heatmap)

        # If U-shaped heatmap is enabled, adjust the heatmap weight distribution
        if u_shape_heatmap:
            # Improved U-shape weight computation (dynamic Gaussian kernel)
            h, w = heatmap.shape
            y_center, x_center = h//2, w//2
            y, x = np.ogrid[:h, :w]

            # Dynamically adjust parameters (based on image size)
            sigma_y = max(h/6, 1)
            sigma_x = max(w/6, 1)

            # Gaussian decay function
            u_weight = np.exp(-((y - y_center)**2/(2*sigma_y**2) +
                              (x - x_center)**2/(2*sigma_x**2)))

            # Apply attention-guided enhancement
            if 'attention_weights' in locals():
                # Upsample the attention weights to image size
                att_map = F.interpolate(attention_weights.unsqueeze(0),
                                      size=(h,w),
                                      mode='bilinear',
                                      align_corners=False)
                # Combine with attention weights
                u_weight = u_weight * (1 + att_map.cpu().numpy()[0])

            # Normalization
            u_weight = (u_weight - u_weight.min()) / (u_weight.max() - u_weight.min() + 1e-8)
            heatmap = np.uint8(heatmap * u_weight)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Ensure OpenCV image and heatmap sizes match
        h, w, _ = cv_image.shape
        heatmap = cv2.resize(heatmap, (w, h))

        # Blend the images (ensure correct channel order)
        superimposed_img = cv2.addWeighted(cv_image, 0.6, heatmap, 0.4, 0)

        save_path = os.path.join(output_dir, f"cam_{filename}")
        cv2.imwrite(save_path, superimposed_img)
        print(f"Saved: {save_path}")

# ===== 4. Execute processing =====
if __name__ == "__main__":
    image_dir = "outputs/deeppcb-infer/relitu"
    output_dir = "outputs/deeppcb-infer/result"
    target_class = 0  # Ensure target_class is within 0 <= target_class < 6
    u_shape_heatmap = True  # Enable U-shaped heatmap

    generate_heatmaps(image_dir, output_dir, target_class=target_class, u_shape_heatmap=u_shape_heatmap)
