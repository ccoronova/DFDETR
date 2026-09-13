"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

from typing import Iterable
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Avoid requiring a graphical interface
import matplotlib.pyplot as plt
import json
import traceback
import math
import sys
import os

import torch
from src.misc import dist

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)

# Add a JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)

    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Add a gradient accumulation factor to simulate a larger batch size
        accumulate_grad_steps = kwargs.get('accumulate_grad_steps', 2)  # Accumulate 2 steps by default
        do_optimizer_step = kwargs.get('_accumulation_step', 0) % accumulate_grad_steps == 0

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            # Scale the loss by the gradient accumulation factor
            loss = loss / accumulate_grad_steps
            scaler.scale(loss).backward()

            if do_optimizer_step:
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            if do_optimizer_step:
                scaler.step(optimizer)
                scaler.update()
                # Update the counter
                kwargs['_accumulation_step'] = 0
            else:
                # Increase the accumulation counter
                kwargs['_accumulation_step'] = kwargs.get('_accumulation_step', 0) + 1
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        outputs = model(samples)

        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict
        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)
        # results = postprocessors(outputs, targets)

        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name
        #     panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator


def save_and_visualize_eval_results(coco_evaluator, stats, output_dir):
    """
    Save evaluation results and generate visualization charts
    - Generate a PR curve chart for all classes (IoU=0.50)
    - Overlay the average PR curve across all classes
    Output location: {output_dir}/eval_visualizations/all_classes_pr_curve.png
    """
    if not dist.is_main_process():
        return

    output_dir = Path(output_dir)
    eval_dir = output_dir / "eval_visualizations"
    eval_dir.mkdir(exist_ok=True, parents=True)

    if "bbox" not in coco_evaluator.coco_eval:
        return

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        eval_data = coco_evaluator.coco_eval["bbox"].eval
        precisions = eval_data['precision']  # [T, R, K, A, M]
        params = coco_evaluator.coco_eval["bbox"].params
        cat_ids = params.catIds
        recalls = params.recThrs  # [R] 0:.01:1

        # Get the class names
        from src.data.coco.coco_dataset import mscoco_category2name
        fallback_names = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

        # Only take IoU=0.50 (index 0), area=all (index 0), maxDet=100 (index 2)
        iou_index = 0
        area_index = 0
        maxdet_index = 2

        class_names = []
        per_class_curves = []
        ap50_values = []

        for idx, catId in enumerate(cat_ids):
            # Get the precision curve [R] for this class
            curve = precisions[iou_index, :, idx, area_index, maxdet_index]

            # Handle invalid values (-1)
            valid_mask = curve > -1
            if not np.any(valid_mask):
                # Fill with NaN if all values are invalid
                curve_processed = np.full_like(curve, np.nan)
                ap50 = 0.0
            else:
                curve_processed = np.where(valid_mask, curve, np.nan)
                ap50 = np.nanmean(curve_processed)

            name = mscoco_category2name.get(catId, fallback_names[idx] if idx < len(fallback_names) else f"Class{catId}")
            class_names.append(name)
            per_class_curves.append(curve_processed)
            ap50_values.append(ap50)

        per_class_curves = np.array(per_class_curves) # [K, R]

        # Compute the mean curve across all classes (Mean PR Curve)
        # Average ignoring NaN
        mean_curve = np.nanmean(per_class_curves, axis=0)

        # Get the overall mAP@0.50
        bbox_stats = coco_evaluator.coco_eval["bbox"].stats
        map50 = bbox_stats[1] if bbox_stats is not None else 0.0

        # --- Plotting ---
        plt.figure(figsize=(8, 8))

        # Use the tab20 colormap to support more distinct class colors
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i % 20) for i in range(len(class_names))]

        # Plot the curve for each class
        for i, (name, curve, ap) in enumerate(zip(class_names, per_class_curves, ap50_values)):
            # Skip if the entire curve is NaN
            if np.all(np.isnan(curve)):
                continue

            plt.plot(recalls, curve, label=f'{name} (AP={ap:.3f})',
                     color=colors[i], linewidth=1.5, alpha=0.7)

        # Plot the average curve (Overall)
        plt.plot(recalls, mean_curve, label=f'All Classes (mAP@0.50={map50:.3f})',
                 color='b', linewidth=3, linestyle='-')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve @ IoU=0.50 (All Classes)')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.05)

        plt.gca().set_aspect('equal', adjustable='box')
        # Place the legend outside or resize the layout to avoid overlap
        if len(class_names) > 10:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust the layout to leave room for the legend
        else:
            plt.legend(loc='best')
            plt.tight_layout()

        save_path = eval_dir / "all_classes_pr_curve.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Combined PR curve generated: {save_path}")

        # Save a brief summary of AP results
        with open(eval_dir / "ap_results.json", "w") as f:
            json.dump({name: {'AP@0.50': ap} for name, ap in zip(class_names, ap50_values)}, f, indent=4, cls=NumpyEncoder)

    except Exception as e:
        print(f"Failed to plot: {e}")
        import traceback
        traceback.print_exc()