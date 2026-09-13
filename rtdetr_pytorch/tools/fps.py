import os
import sys
import time
from pathlib import Path

import torch

# Ensure project root is on sys.path (Windows-friendly)
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent  # .../rtdetr_pytorch
sys.path.insert(0, str(_PROJECT_ROOT))

from src.nn.backbone import PResNet
from src.zoo.rtdetr import HybridEncoder, RTDETRTransformer, RTDETR


# --------- User-adjustable parameters ---------
WEIGHTS_PATH = 'outputs/best_map50.pth'
INPUT_SIZE = 640
NUM_CLASSES = 6
BATCH_SIZE = 8
WARMUP_ITERS = 50
ITERATIONS = 100
USE_AMP = True


def _extract_state_dict(ckpt: object) -> dict:
    """Best-effort extraction of state_dict from common checkpoint formats."""
    if isinstance(ckpt, dict):
        # Common save formats: {'model': ...} or {'ema': {'module': ...}} etc.
        if 'ema' in ckpt and ckpt['ema'] is not None:
            ema = ckpt['ema']
            if isinstance(ema, dict) and 'module' in ema and isinstance(ema['module'], dict):
                return ema['module']
            if isinstance(ema, dict):
                return ema
        for k in ("model", "state_dict", "net"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        # It may be the state_dict directly
        if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
        return ckpt
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def build_rtdetr_model(input_size: int = INPUT_SIZE, num_classes: int = NUM_CLASSES) -> torch.nn.Module:
    """Build RT-DETR according to the network structure of your current project."""
    backbone = PResNet(depth=18, freeze_norm=False, pretrained=False, return_idx=[1, 2, 3])

    # Note: the HybridEncoder implementation relies on eval_spatial_size to build position encodings
    encoder = HybridEncoder(
        in_channels=[128, 256, 512],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act='gelu',
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=0.5,
        act='silu',
        eval_spatial_size=[input_size, input_size],
    )

    decoder = RTDETRTransformer(
        num_classes=num_classes,
        eval_idx=-1,
        num_decoder_layers=3,
        num_denoising=100,
        feat_channels=[256] * 3,
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        num_levels=3,
        num_queries=300,
        eval_spatial_size=[input_size, input_size],
    )

    model = RTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        multi_scale=[input_size],
    )

    return model


def load_weights(model: torch.nn.Module, weights_path: str) -> None:
    ckpt = torch.load(weights_path, map_location="cpu")
    sd = _extract_state_dict(ckpt)
    incompatible = model.load_state_dict(sd, strict=False)

    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))

    print(f"Loaded weights: {weights_path}")
    print(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if missing:
        print("Missing keys (sample):")
        for k in missing[:30]:
            print(f"  - {k}")
    if unexpected:
        print("Unexpected keys (sample):")
        for k in unexpected[:30]:
            print(f"  - {k}")


def try_get_flops_params(model: torch.nn.Module, input_res: tuple[int, int, int]) -> tuple[str, str]:
    """Return (flops, params) as strings. If ptflops not available, return ('N/A','N/A')."""
    try:
        from ptflops import get_model_complexity_info  # type: ignore
    except Exception:
        return "N/A (install ptflops)", "N/A (install ptflops)"

    try:
        flops, params = get_model_complexity_info(
            model,
            input_res,
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
        return flops, params
    except Exception as e:
        return f"N/A ({e})", f"N/A ({e})"


def benchmark(
    weights_path: str = WEIGHTS_PATH,
    input_size: int = INPUT_SIZE,
    batch_size: int = BATCH_SIZE,
    warmup_iters: int = WARMUP_ITERS,
    iterations: int = ITERATIONS,
    use_amp: bool = USE_AMP,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # perf knobs
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_rtdetr_model(input_size=input_size, num_classes=NUM_CLASSES)
    load_weights(model, weights_path)

    model.to(device)
    model.eval()

    # FLOPs / Params (may be N/A)
    flops, params = try_get_flops_params(model, (3, input_size, input_size))

    # CPU memory
    try:
        import psutil  # type: ignore

        initial_cpu_mem = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        psutil = None
        initial_cpu_mem = 0.0

    # GPU memory init
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    x = torch.randn(batch_size, 3, input_size, input_size, device=device)

    # warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            if use_amp and device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _ = model(x)
            else:
                _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # timed iterations (CUDA events for accuracy)
    times_ms: list[float] = []
    with torch.no_grad():
        if torch.cuda.is_available():
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            for _ in range(iterations):
                starter.record()
                if use_amp and device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        _ = model(x)
                else:
                    _ = model(x)
                ender.record()
                torch.cuda.synchronize()
                times_ms.append(starter.elapsed_time(ender))
        else:
            for _ in range(iterations):
                t0 = time.perf_counter()
                _ = model(x)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

    mean_time_ms = float(sum(times_ms) / len(times_ms)) if times_ms else float("nan")
    fps = (1000.0 / mean_time_ms) * batch_size if mean_time_ms > 0 else float("inf")

    # CPU memory end
    if 'psutil' in locals() and psutil is not None:
        final_cpu_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        cpu_mem_usage = final_cpu_mem - initial_cpu_mem
    else:
        cpu_mem_usage = float('nan')

    # GPU memory
    if torch.cuda.is_available():
        max_gpu_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        reserved_gpu_mem = torch.cuda.memory_reserved() / (1024 * 1024)
    else:
        max_gpu_mem = 0.0
        reserved_gpu_mem = 0.0

    # Output & save
    lines = []
    lines.append("=" * 60)
    lines.append("RT-DETR Benchmark")
    lines.append("=" * 60)
    lines.append(f"Weights: {weights_path}")
    lines.append(f"Device: {device}")
    lines.append(f"Input: ({batch_size}, 3, {input_size}, {input_size})")
    lines.append(f"AMP: {use_amp}")
    lines.append(f"Warmup iters: {warmup_iters}, Test iters: {iterations}")
    lines.append(f"FLOPs: {flops}")
    lines.append(f"Params: {params}")
    lines.append(f"Avg Inference Time: {mean_time_ms:.6f} ms")
    lines.append(f"FPS: {fps:.2f}")
    if cpu_mem_usage == cpu_mem_usage:
        lines.append(f"CPU Memory Usage: {cpu_mem_usage:.2f} MB")
    else:
        lines.append("CPU Memory Usage: N/A (install psutil)")
    lines.append(f"GPU Memory Usage (Max Allocated): {max_gpu_mem:.2f} MB")
    lines.append(f"GPU Memory Usage (Reserved): {reserved_gpu_mem:.2f} MB")

    print("\n".join(lines))

    try:
        save_dir = Path(weights_path).resolve().parent
        save_path = save_dir / "rtdetr_benchmark.txt"
        save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nResults saved to: {save_path}")
    except Exception as e:
        print(f"Failed to save results: {e}")


if __name__ == "__main__":
    benchmark()
