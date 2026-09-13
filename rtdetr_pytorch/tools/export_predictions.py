# -*- coding: utf-8 -*-
"""
Run inference with RT-DETR family models (baseline / SurfDETR / ablation variants) on the validation
set and export a COCO-format predictions.json.
Reused by texture_group_eval.py / collect_all_metrics.py. Pure inference, no training.

Usage (inside the rtdetr_pytorch directory):
  python tools/export_predictions.py --dataset pcb --ckpt <checkpoint path> --out <output json>
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

# Model checkpoint presets (example paths). Each model's corresponding Encoder enhancement switches are as follows:
#   SurfDETR (old AMSF) / ABL-* / RT-DETR / DFDETR (default), see build_model.
CKPTS = {
    ("pcb", "SurfDETR"): os.path.join(REPO_ROOT, "outputs", "pcb", "best_map50.pth"),
    ("deeppcb", "SurfDETR"): os.path.join(REPO_ROOT, "outputs", "deeppcb", "best_map50.pth"),
    # Pure RT-DETR baseline
    ("pcb", "RT-DETR"): os.path.join(REPO_ROOT, "outputs", "pcb", "baseline_map50.pth"),
    ("deeppcb", "RT-DETR"): os.path.join(REPO_ROOT, "outputs", "deeppcb", "baseline_map50.pth"),
    # Ablation variants
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
    # Drill down to the real state_dict layer by layer: ema may be {"module": ...}; model is the weights directly
    sd = ck
    for key in ("ema", "model"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
            # ema has one more "module" layer underneath
            if isinstance(sd, dict) and "module" in sd:
                sd = sd["module"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"    Loaded weights: missing={len(missing)} unexpected={len(unexpected)}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["pcb", "deeppcb"], required=True)
    ap.add_argument("--model", default=None, help="Model name in CKPTS, e.g. RT-DETR / SurfDETR")
    ap.add_argument("--ckpt", default=None, help="Or directly give a checkpoint path")
    ap.add_argument("--out", default=None, help="Output predictions.json path")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    ckpt = args.ckpt or CKPTS.get((args.dataset, args.model))
    if not ckpt or not os.path.exists(ckpt):
        raise SystemExit(f"Weights not found or not configured: dataset={args.dataset} model={args.model} ckpt={ckpt}")
    out = args.out or os.path.join(os.path.dirname(ckpt), "predictions_baseline.json" if args.model == "RT-DETR" else "predictions.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.model or ckpt}  Dataset: {args.dataset}  Device: {device}")

    model_name = args.model or ""
    # Enhancement module switches per model
    #   RT-DETR: all off (pure baseline);  legacy SurfDETR/ABL-*: only AMSF;  DFDETR (default): only DADC+SWFD
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
        # postprocessor expects orig_target_sizes=[width, height] (x multiplies dim 0, y multiplies dim 1)
        res = post(outs, torch.tensor([[w0, h0]]).to(device))
        r0 = res[0]  # {"labels","boxes","scores"}
        for l, b, s in zip(r0["labels"].cpu().numpy(), r0["boxes"].cpu().numpy(), r0["scores"].cpu().numpy()):
            if s < 0.001:  # keep very low scores so COCOeval uses the full prediction distribution
                continue
            x1, y1, x2, y2 = [float(v) for v in b]
            results.append({
                "image_id": int(img_id),
                "category_id": int(l) + 1,  # model 0-based -> ground truth 1-based
                "bbox": [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)],
                "score": round(float(s), 5),
            })
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(files)}")

    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"Done: {len(results)} predictions -> {out}")


if __name__ == "__main__":
    main()
