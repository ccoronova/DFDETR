import torch
import yaml
import time
from src.zoo.rtdetr import RTDETR  # RTDETR 模型
from src.nn.backbone import PResNet
from src.zoo.rtdetr import HybridEncoder, RTDETRTransformer
from thop import profile, clever_format  # 导入 thop 用于计算 FLOPs

# 加载模型
def load_model(config_path, device='cuda'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 解析配置文件，获取模型配置字典
    backbone_cfg = config['PResNet']
    encoder_cfg = config['HybridEncoder']
    decoder_cfg = config['RTDETRTransformer']
    multi_scale = config['RTDETR']['multi_scale']

    # 实例化模型组件
    backbone = PResNet(**backbone_cfg)
    encoder = HybridEncoder(**encoder_cfg)
    decoder = RTDETRTransformer(**decoder_cfg)

    # 创建 RTDETR 模型实例
    model = RTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        multi_scale=multi_scale,
    )
    model.to(device)
    model.eval()
    return model

# 打印模型的参数数量
def print_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {total_params / 1e6:.2f} M")  # 以百万为单位输出

# 打印 FLOPs 和参数数量
def print_model_flops(model, input_data):
    try:
        macs, params = profile(model, inputs=(input_data,), verbose=False)
        macs, params = clever_format([macs, params], "%.3f")
        print(f"模型 FLOPs: {macs}, 参数数量: {params}")
    except Exception as e:
        print(f"无法计算 FLOPs 或参数数量，错误信息: {e}")

# 计算 FPS
def calculate_fps(model, input_data, device, iterations=100):
    model.eval()
    input_data = input_data.to(device)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_data)
    end_time = time.time()
    fps = iterations / (end_time - start_time)
    print(f"FPS: {fps:.2f} 帧/秒")
    return fps

# 主函数
if __name__ == "__main__":
    # 配置文件路径
    config_path = "configs/rtdetr/rtdetr_r18vd_6x_coco.yml"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型
    model = load_model(config_path, device)

    # 假设您有 3 个输入特征图，可以动态地模拟不同通道数的输入
    # 例如，假设第一个特征图通道数为 64，第二个为 128，第三个为 256
    feature_map_1 = torch.randn(1, 64, 640, 640)  # 第一个特征图
    feature_map_2 = torch.randn(1, 128, 640, 640)  # 第二个特征图
    feature_map_3 = torch.randn(1, 256, 640, 640)  # 第三个特征图

    # 合并输入特征图
    dummy_input = torch.cat([feature_map_1, feature_map_2, feature_map_3], dim=1).to(device)  # 将所有特征图沿通道维度拼接

    # 打印模型参数数量
    print_model_params(model)

    # 打印模型 FLOPs (GLOPS)
    print_model_flops(model, dummy_input)

    # 计算并显示 FPS
    calculate_fps(model, dummy_input, device)
