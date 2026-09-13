# -*- coding: utf-8 -*-
"""
对 RT-DETR 系模型(基线 / SurfDETR / 消融变体)在验证集上推理,导出 COCO 格式 predictions.json。
供 texture_group_eval.py / collect_all_metrics.py 复用。纯推理,无训练。

用法(rtdetr_pytorch 目录下):
  python tools/export_predictions.py --dataset pcb --ckpt <checkpoint路径> --out <输出json>
"""
import os
import sys
import json
import argparse

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))

from src.zoo.rtdetr import RTDETR, HybridEncoder, RTDETRTransformer, RTDETRPostProcessor  # noqa: E402
from src.nn.backbone import PResNet  # noqa: E402

IMGSZ = 640

DATASETS = {
    "pcb": {
        "num_classes": 6,
        "img_dir": os.path.join(REPO_ROOT, "PCB_DATASET", "val"),
        "ann_file": os.path.join(REPO_ROOT, "PCB_DATASET", "val.json"),
    },
    "deeppcb": {
        "num_classes": 6,
        "img_dir": os.path.join(REPO_ROOT, "Deeppcb", "dataset_coco", "val", "images"),
        "ann_file": os.path.join(REPO_ROOT, "Deeppcb", "dataset_coco", "annotations", "val_coco_format.json"),
    },
}

# 模型 checkpoint 预设(示例路径)。各模型对应 Encode 增强开关如下:
#   SurfDETR(旧版 AMSF) / ABL-* / RT-DETR / DFDETR(默认),见 build_model。
CKPTS = {
    ("pcb", "SurfDETR"): os.path.join(REPO_ROOT, "outputs", "pcb", "best_map50.pth"),
    ("deeppcb", "SurfDETR"): os.path.join(REPO_ROOT, "outputs", "deeppcb", "best_map50.pth"),
    # 纯 RT-DETR 基线
    ("pcb", "RT-DETR"): os.path.join(REPO_ROOT, "outputs", "pcb", "baseline_map50.pth"),
    ("deeppcb", "RT-DETR"): os.path.join(REPO_ROOT, "outputs", "deeppcb", "baseline_map50.pth"),
    # 消融变体
    ("pcb", "ABL-full"): os.path.join(REPO_ROOT, "outputs", "pcb", "abl_full.pth"),
    ("pcb", "ABL-TAM"): os.path.join(REPO_ROOT, "outputs", "pcb", "abl_tam.pth"),
}


def build_model(num_classes, use_amsf=False, use_dadc=True, use_swfd=True):
    backbone = PResNet(depth=18, freeze_norm=False, pretrained=False, return_idx=[1, 2, 3])
    encoder = HybridEncoder(in_channels=[128, 256, 512], enc_act="gelu", expansion=0.5,
                            eval_spatial_size=[IMGSZ, IMGSZ],
                            use_amsf=use_amsf,
                            use_edge_enhancer=use_amsf, use_texture_aware=use_amsf,
                            use_dadc=use_dadc, use_swfd=use_swfd)
    decoder = RTDETRTransformer(num_classes=num_classes, hidden_dim=256, num_decoder_layers=3,
                                num_denoising=100, feat_channels=[256, 256, 256],
                                feat_strides=[8, 16, 32], eval_spatial_size=[IMGSZ, IMGSZ])
    return RTDETR(backbone=backbone, encoder=encoder, decoder=decoder, multi_scale=[IMGSZ])


def load_ckpt(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    # 逐层取到真正的 state_dict: ema 可能是 {"module":...}; model 直接是权重
    sd = ck
    for key in ("ema", "model"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
            # ema 下还有一层 "module"
            if isinstance(sd, dict) and "module" in sd:
                sd = sd["module"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"    加载权重: missing={len(missing)} unexpected={len(unexpected)}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["pcb", "deeppcb"], required=True)
    ap.add_argument("--model", default=None, help="CKPTS 里的模型名,如 RT-DETR / SurfDETR")
    ap.add_argument("--ckpt", default=None, help="或直接给 checkpoint 路径")
    ap.add_argument("--out", default=None, help="输出 predictions.json 路径")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    ckpt = args.ckpt or CKPTS.get((args.dataset, args.model))
    if not ckpt or not os.path.exists(ckpt):
        raise SystemExit(f"权重不存在或未配置: dataset={args.dataset} model={args.model} ckpt={ckpt}")
    out = args.out or os.path.join(os.path.dirname(ckpt), "predictions_baseline.json" if args.model == "RT-DETR" else "predictions.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"模型:{args.model or ckpt}  数据集:{args.dataset}  设备:{device}")

    model_name = args.model or ""
    # 各模型的增强模块开关
    #   RT-DETR: 全关（纯基线）;  legacy SurfDETR/ABL-*: 仅 AMSF;  DFDETR(默认): 仅 DADC+SWFD
    if model_name == "RT-DETR":
        use_amsf, use_dadc, use_swfd = False, False, False
    elif any(t in model_name for t in ("SurfDETR", "ABL")):
        use_amsf, use_dadc, use_swfd = True, False, False
    else:  # DFDETR
        use_amsf, use_dadc, use_swfd = False, True, True
    model = load_ckpt(build_model(cfg["num_classes"], use_amsf, use_dadc, use_swfd), ckpt).to(device).eval()
    post = RTDETRPostProcessor(num_classes=cfg["num_classes"])

    with open(cfg["ann_file"], encoding="utf-8") as f:
        coco = json.load(f)
    # file_name -> image_id
    name2id = {im["file_name"]: im["id"] for im in coco["images"]}
    stem2id = {im["file_name"][: im["file_name"].rfind(".")]: im["id"] for im in coco["images"]}

    tf = transforms.Compose([transforms.Resize((IMGSZ, IMGSZ)), transforms.ToTensor()])
    results = []
    files = sorted(os.listdir(cfg["img_dir"]))
    for i, fn in enumerate(files):
        img_id = name2id.get(fn, stem2id.get(os.path.splitext(fn)[0]))
        if img_id is None:
            continue
        img = Image.open(os.path.join(cfg["img_dir"], fn)).convert("RGB")
        w0, h0 = img.size
        x = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outs = model(x)
        # postprocessor 期望 orig_target_sizes=[宽,高](x乘第0维,y乘第1维)
        res = post(outs, torch.tensor([[w0, h0]]).to(device))
        r0 = res[0]  # {"labels","boxes","scores"}
        for l, b, s in zip(r0["labels"].cpu().numpy(), r0["boxes"].cpu().numpy(), r0["scores"].cpu().numpy()):
            if s < 0.001:  # 保留极低分,让 COCOeval 用完整预测分布
                continue
            x1, y1, x2, y2 = [float(v) for v in b]
            results.append({
                "image_id": int(img_id),
                "category_id": int(l) + 1,  # 模型 0-based -> 真值 1-based
                "bbox": [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)],
                "score": round(float(s), 5),
            })
        if (i + 1) % 20 == 0:
            print(f"  已处理 {i+1}/{len(files)}")

    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"完成: {len(results)} 个预测 -> {out}")


if __name__ == "__main__":
    main()
