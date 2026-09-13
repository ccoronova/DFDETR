import torch
import yaml
import time
from src.zoo.rtdetr import RTDETR  # RTDETR model
from src.nn.backbone import PResNet
from src.zoo.rtdetr import HybridEncoder, RTDETRTransformer
from thop import profile, clever_format  # Import thop to compute FLOPs

# Load model
def load_model(config_path, device='cuda'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Parse the config file to get the model configuration dictionary
    backbone_cfg = config['PResNet']
    encoder_cfg = config['HybridEncoder']
    decoder_cfg = config['RTDETRTransformer']
    multi_scale = config['RTDETR']['multi_scale']

    # Instantiate model components
    backbone = PResNet(**backbone_cfg)
    encoder = HybridEncoder(**encoder_cfg)
    decoder = RTDETRTransformer(**decoder_cfg)

    # Create RTDETR model instance
    model = RTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        multi_scale=multi_scale,
    )
    model.to(device)
    model.eval()
    return model

# Print the model's parameter count
def print_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {total_params / 1e6:.2f} M")  # Output in millions

# Print FLOPs and parameter count
def print_model_flops(model, input_data):
    try:
        macs, params = profile(model, inputs=(input_data,), verbose=False)
        macs, params = clever_format([macs, params], "%.3f")
        print(f"Model FLOPs: {macs}, parameter count: {params}")
    except Exception as e:
        print(f"Unable to compute FLOPs or parameter count, error: {e}")

# Compute FPS
def calculate_fps(model, input_data, device, iterations=100):
    model.eval()
    input_data = input_data.to(device)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_data)
    end_time = time.time()
    fps = iterations / (end_time - start_time)
    print(f"FPS: {fps:.2f} frames/second")
    return fps

# Main function
if __name__ == "__main__":
    # Config file path
    config_path = "configs/rtdetr/rtdetr_r18vd_6x_coco.yml"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = load_model(config_path, device)

    # Assume you have 3 input feature maps; you can simulate inputs with different channel counts
    # For example, assume the first feature map has 64 channels, the second 128, the third 256
    feature_map_1 = torch.randn(1, 64, 640, 640)  # First feature map
    feature_map_2 = torch.randn(1, 128, 640, 640)  # Second feature map
    feature_map_3 = torch.randn(1, 256, 640, 640)  # Third feature map

    # Concatenate input feature maps
    dummy_input = torch.cat([feature_map_1, feature_map_2, feature_map_3], dim=1).to(device)  # Concatenate feature maps along the channel dimension

    # Print model parameter count
    print_model_params(model)

    # Print model FLOPs (GLOPS)
    print_model_flops(model, dummy_input)

    # Compute and display FPS
    calculate_fps(model, dummy_input, device)
