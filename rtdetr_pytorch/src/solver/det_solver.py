'''
by lyuwenyu
'''
import os
import time
import json
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Avoid requiring a graphical interface

import torch

from src.misc import dist
from src.data import get_coco_api_from_dataset
from src.data.coco.coco_dataset import mscoco_category2name, mscoco_label2category

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate, save_and_visualize_eval_results
from calflops import calculate_flops

# Add a custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Helper function: Get the correct class name from index
def get_class_name(idx, catId):
    """
    Get the correct class name

    Args:
        idx: Class index (0-based)
        catId: Category ID returned by COCO evaluator

    Returns:
        str: Class name
    """
    # PCB dataset class names
    pcb_classes = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

    # Method 1: Use idx+1 as key to look up in mscoco_category2name dictionary
    category_id = idx + 1
    if category_id in mscoco_category2name:
        return mscoco_category2name[category_id]

    # Method 2: Try using catId directly
    if catId in mscoco_category2name:
        return mscoco_category2name[catId]

    # Method 3: Use predefined class list
    if idx < len(pcb_classes):
        return pcb_classes[idx]

    # Fallback option
    return f"Class{category_id}"

class DetSolver(BaseSolver):

    def fit(self):
        print("Start training")
        self.train()

        args = self.cfg

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        best_stat = {'epoch': -1, 'best_iou_50': 0.0}  # Track best IoU@0.50

        # Track the best mAP@0.50 value
        best_map50 = 0.0
        best_map50_epoch = -1

        # Store detailed metrics of the best model
        best_map50_details = {}

        # Initialize the loss history lists
        train_loss_history = []
        val_loss_history = []

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            # Train for one epoch
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            # Record training loss
            if 'loss' in train_stats:
                train_loss_history.append(train_stats['loss'])

            # Step the LR scheduler
            self.lr_scheduler.step()

            # Only keep the latest checkpoint, no longer saving multiple checkpoints per step
            if self.output_dir:
                checkpoint_path = self.output_dir / 'checkpoint.pth'
                dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            # Evaluate the model
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            # Record validation loss
            if 'loss' in test_stats:
                val_loss = test_stats['loss']
                if isinstance(val_loss, list):
                     val_loss = val_loss[0]
                val_loss_history.append(val_loss)

            # Track IoU@0.50
            if coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
                iou_50 = coco_evaluator.coco_eval["bbox"].stats[1]  # Index 1 corresponds to IoU@0.50
                map50_95 = coco_evaluator.coco_eval["bbox"].stats[0]  # Index 0 corresponds to mAP@0.50-0.95

                print(f"Epoch {epoch}, IoU@0.50: {iou_50:.4f}, mAP@0.50-0.95: {map50_95:.4f}")

                # Update the best IoU@0.50 and corresponding epoch
                if iou_50 > best_stat['best_iou_50']:
                    best_stat['best_iou_50'] = iou_50
                    best_stat['epoch'] = epoch
                    print(f"New best IoU@0.50: {best_stat['best_iou_50']:.4f} at Epoch {best_stat['epoch']}")

                # Track the best mAP@0.50 value
                if iou_50 > best_map50:
                    best_map50 = iou_50
                    best_map50_epoch = epoch
                    # Save the best mAP@0.50 model
                    if self.output_dir and dist.is_main_process():
                        best_map50_path = self.output_dir / 'best_map50.pth'
                        dist.save_on_master(self.state_dict(epoch), best_map50_path)
                        print(f"Best mAP@0.50 model saved, metric: {best_map50:.4f}, epoch: {best_map50_epoch}")

                        # Save the best evaluator for later visualization
                        best_evaluator_path = self.output_dir / 'best_coco_evaluator.pth'
                        torch.save(coco_evaluator, best_evaluator_path)

                        # Save the detailed metrics of the best model
                        best_map50_details = {
                            'epoch': epoch,
                            'iou_50': iou_50,
                            'map_50_95': map50_95,
                            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'other_metrics': {k: v for k, v in test_stats.items() if k not in ['loss', 'time']},
                            'class_metrics': {}
                        }

                        # Add detailed metrics for each class
                        try:
                            precisions = coco_evaluator.coco_eval["bbox"].eval['precision']
                            cat_ids = coco_evaluator.coco_eval["bbox"].params.catIds

                            # Only save the needed evaluation metrics: AP@0.50-0.95 (index 0) and AP@0.50 (index 1)
                            all_stats = coco_evaluator.coco_eval["bbox"].stats.tolist()
                            # Only save the two metrics we care about
                            best_map50_details['coco_stats'] = [all_stats[0], all_stats[1]]

                            for idx, catId in enumerate(cat_ids):
                                # AP@0.50
                                ap50 = precisions[0, :, idx, 0, 2]
                                ap50 = ap50[ap50 > -1]
                                ap50_score = float(ap50.mean()) if ap50.size > 0 else float('nan')

                                # AP@0.50-0.95
                                ap_all = precisions[:, :, idx, 0, 2]
                                ap_all = ap_all[ap_all > -1]
                                ap_all_score = float(ap_all.mean()) if ap_all.size > 0 else float('nan')

                                # Get the class name
                                label = mscoco_category2name.get(catId, str(catId))

                                # Store in the details
                                best_map50_details['class_metrics'][label] = {
                                    'id': catId,
                                    'AP@0.50': ap50_score,
                                    'AP@0.50-0.95': ap_all_score
                                }
                        except Exception as e:
                            print(f"Error saving per-class metric details: {e}")
                            # traceback.print_exc()

                        # Ensure the directory exists
                        os.makedirs(self.output_dir, exist_ok=True)

                        # Save in JSON format
                        try:
                            json_path = self.output_dir / "best_map50_details.json"
                            with open(json_path, "w") as f:
                                json.dump(best_map50_details, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
                        except Exception as e:
                            print(f"Error saving JSON metric details: {e}")

                        # Also create an easy-to-read txt version
                        try:
                            txt_path = self.output_dir / "best_map50_details.txt"
                            with open(txt_path, "w") as f:
                                f.write(f"===== Best mAP@0.50 model details =====\n\n")
                                f.write(f"Training epoch: {epoch}\n")
                                f.write(f"Saved date: {best_map50_details['date']}\n")
                                f.write(f"Model path: {self.output_dir / 'best_map50.pth'}\n\n")

                                f.write("===== Overall performance metrics =====\n")
                                f.write(f"mAP@0.50: {iou_50:.4f}\n\n")

                                # Per-class performance table header
                                f.write("===== Per-class performance metrics =====\n")
                                f.write("| {:<20} | {:<10} |\n".format("Class Name", "AP@0.50"))
                                f.write("|" + "-"*22 + "|" + "-"*12 + "|\n")

                                # Per-class performance data rows
                                if best_map50_details['class_metrics']:
                                    for label, metrics in sorted(best_map50_details['class_metrics'].items()):
                                        if label and metrics and 'AP@0.50' in metrics:
                                            ap50 = metrics['AP@0.50']
                                            if not math.isnan(ap50):
                                                f.write("| {:<20} | {:<10.4f} |\n".format(
                                                    label[:20], ap50))
                                else:
                                    # Add a notice if there are no per-class metrics
                                    f.write("| No per-class metric data - the model may still be in early training |\n")

                        except Exception as e:
                            print(f"Error saving TXT metric details: {e}")

                # Track the best mAP@0.50-0.95 value
                # The best_map50_95 saving logic has been removed

            # TODO
            for k in test_stats.keys():
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
            print('best_stat: ', best_stat)


            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs - only save the latest evaluation result
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                  self.output_dir / "eval" / 'latest.pth')

                        # Also save the evaluation result of the best mAP model
                        if iou_50 == best_map50:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                      self.output_dir / "eval" / 'best_map50.pth')

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        # After training, specifically load the best model and evaluate it to generate its evaluation results
        # First evaluate the best mAP@0.50 model
        if dist.is_main_process() and best_map50_epoch >= 0:
            print("\n" + "="*60)
            print("Loading the best mAP@0.50 model and re-evaluating to generate the best model's visualization...")
            best_map50_path = self.output_dir / 'best_map50.pth'

            if best_map50_path.exists():
                # Save the current model state
                current_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

                try:
                    # Load the best mAP@0.50 model
                    best_state_dict = torch.load(best_map50_path, map_location=self.device)

                    print(f"Loading the best mAP@0.50 model (Epoch {best_map50_epoch})...")
                    # Prefer the EMA branch for re-evaluation to stay consistent with training-time evaluation
                    loaded_branch = None
                    if 'ema' in best_state_dict and best_state_dict['ema'] is not None:
                        ema_state = best_state_dict['ema']
                        if isinstance(ema_state, dict) and 'module' in ema_state:
                            self.model.load_state_dict(ema_state['module'])
                            loaded_branch = 'EMA(module)'
                        else:
                            self.model.load_state_dict(ema_state)
                            loaded_branch = 'EMA'
                    elif 'model' in best_state_dict:
                        self.model.load_state_dict(best_state_dict['model'])
                        loaded_branch = 'MODEL'
                    else:
                        self.model.load_state_dict(best_state_dict)
                        loaded_branch = 'RAW_STATE_DICT'
                    print(f"Re-evaluating with {loaded_branch} weights to stay consistent with the training phase.")

                    # Create a dedicated directory to store the best model's evaluation results
                    best_map50_eval_dir = self.output_dir / "best_map50_eval"
                    best_map50_eval_dir.mkdir(exist_ok=True)

                    print(f"Evaluating with the best mAP@0.50 model; results will be saved to {best_map50_eval_dir}...")

                    # Re-evaluate the best model
                    base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
                    test_stats, best_coco_evaluator = evaluate(
                        self.model, self.criterion, self.postprocessor,
                        self.val_dataloader, base_ds, self.device, best_map50_eval_dir
                    )
                    # Print alignment: compare the re-evaluated mAP@0.50 with the best value recorded during training
                    try:
                        if best_coco_evaluator is not None and "bbox" in best_coco_evaluator.coco_eval:
                            reeval_map50 = float(best_coco_evaluator.coco_eval["bbox"].stats[1])
                            print(f"Re-evaluated mAP@0.50: {reeval_map50:.4f} (training best: {best_map50:.4f})")
                    except Exception:
                        pass

                    # Save the evaluation result; it will be processed by save_and_visualize_eval_results in det_engine.py
                    # to generate the corresponding PR curve and evaluation charts, saved under best_map50_eval
                    if best_coco_evaluator is not None and "bbox" in best_coco_evaluator.coco_eval:
                        torch.save(
                            best_coco_evaluator.coco_eval["bbox"].eval,
                            best_map50_eval_dir / "best_map50_eval_result.pth"
                        )
                        print(f"The evaluation result of the best mAP@0.50 model has been saved")
                        # New: save the PR curve and ap_results.json
                        from src.solver.det_engine import save_and_visualize_eval_results
                        print(f"Generating the best model's visualization results...")
                        try:
                            save_and_visualize_eval_results(
                                best_coco_evaluator, test_stats, best_map50_eval_dir
                            )
                            print(f"Visualization results of the best mAP@0.50 model generated at: {best_map50_eval_dir / 'eval_visualizations'}")
                        except Exception as viz_error:
                            print(f"Error generating visualization: {viz_error}")
                            import traceback
                            traceback.print_exc()

                        # Copy the generated visualization files to the main visualization directory for unified viewing
                        # vis_source_dir = best_map50_eval_dir / "eval_visualizations"
                        # vis_target_dir = self.output_dir / "visualizations" / "best_map50"
                        # if vis_source_dir.exists():
                        #     vis_target_dir.mkdir(exist_ok=True, parents=True)
                        #     import shutil
                        #     for vis_file in vis_source_dir.glob("*"):
                        #         if vis_file.is_file():
                        #             shutil.copy2(vis_file, vis_target_dir / vis_file.name)
                        #     print(f"Visualization results of the best mAP@0.50 model copied to {vis_target_dir}")

                    # Restore the model to its previous state
                    self.model.load_state_dict(current_model_state)
                    print("Model state restored to the state at the end of training")
                except Exception as e:
                    print(f"Error loading and evaluating the best mAP@0.50 model: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Warning: the best mAP@0.50 model file does not exist: {best_map50_path}")

        # Print the best model metric summary
        print("\n" + "="*60)
        print("               Training finished, best model summary              ")
        print("="*60)

        # Load the saved best model details
        best_map50_file = self.output_dir / "best_map50_details.json"

        # Load the detailed metric data (if it exists)
        best_map50_data = {}

        if best_map50_file.exists() and dist.is_main_process():
            try:
                with open(best_map50_file, "r") as bf:
                    best_map50_data = json.load(bf)
            except Exception as e:
                print(f"Error reading the best mAP@0.50 metric file: {e}")

        # Print the best mAP@0.50 model info
        print("\n[Best mAP@0.50 Model]")
        print("-"*60)
        print(f"Metric value: {best_map50:.4f}")
        print(f"Training epoch: {best_map50_epoch}")
        print(f"Model path: {self.output_dir / 'best_map50.pth'}")

        # Print the per-class metric table
        if "class_metrics" in best_map50_data and best_map50_data["class_metrics"]:
            print("\nPer-class AP values:")
            print("{:<20} | {:<10}".format("Class Name", "AP@0.50"))
            print("-" * 33)

            for cls_name, metrics in sorted(best_map50_data["class_metrics"].items()):
                ap50 = metrics.get('AP@0.50', float('nan'))
                print("{:<20} | {:<10.4f}".format(cls_name[:20], ap50))

        print("\n" + "-"*60)
        print(f"Final checkpoint path: {self.output_dir / 'checkpoint.pth'}")
        print("="*60)

        # Create a summary file containing all metrics of the best model at the end of training
        # Create a summary file containing all metrics of the best model at the end of training
        if dist.is_main_process() and self.output_dir:
            # Ensure the output directory exists (self.output_dir is a Path)
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Use a timestamp to create a unique summary filename
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            summary_file = self.output_dir / f"training_summary_{timestamp}.txt"

            try:
                # Create the summary file
                with open(summary_file, "w", encoding="utf-8") as f:
                    f.write("="*60 + "\n")
                    f.write("                 RT-DETR Training Summary                 \n")
                    f.write("="*60 + "\n\n")

                    # Basic training info
                    f.write("[Training Basic Info]\n")
                    f.write(f"- Completion time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"- Epochs: {args.epoches}\n")
                    f.write(f"- Training duration: {total_time_str}\n")
                    f.write(f"- Model parameters: {n_parameters:,}\n")

                    # Compute the model's computational cost
                    batch_size = 1
                    input_shape = (batch_size, 3, 640, 640)
                    flops, macs, params = calculate_flops(
                        model=self.model,
                        input_shape=input_shape,
                        output_as_string=True,
                        output_precision=4,
                        print_detailed=False
                    )
                    f.write(f"- FLOPs: {flops}\n")
                    f.write(f"- MACs: {macs}\n")
                    f.write(f"- Params: {params}\n")
                    f.write(f"- Output directory: {self.output_dir}\n\n")

                    # Best mAP@0.50 model
                    f.write("-"*60 + "\n")
                    f.write("[Best mAP@0.50 Model]\n")
                    f.write("-"*60 + "\n")
                    f.write(f"- Metric value: {best_map50:.4f}\n")
                    f.write(f"- Epoch: {best_map50_epoch}\n")
                    f.write(f"- Model path: {self.output_dir / 'best_map50.pth'}\n")
                    if best_map50_data:
                        f.write(f"- Saved date: {best_map50_data.get('date', 'Not recorded')}\n")

                    # If per-class metrics exist, add detailed info in table form
                    if "class_metrics" in best_map50_data and best_map50_data["class_metrics"]:
                        f.write("\n[Per-class AP Values]\n")
                        f.write("| {:<20} | {:<10} |\n".format("Class Name", "AP@0.50"))
                        f.write("|" + "-"*22 + "|" + "-"*12 + "|\n")

                        for cls_name, metrics in sorted(best_map50_data["class_metrics"].items()):
                            ap50 = metrics.get('AP@0.50', float('nan'))
                            f.write("| {:<20} | {:<10.4f} |\n".format(cls_name[:20], ap50))
                    f.write("\n\n" + "="*60 + "\n")
                    f.write("Training summary generated on: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
                    f.write("="*60 + "\n")

                print(f"\nTraining summary saved to: {summary_file}")

                # Create a copy of the latest training summary (instead of a symlink, more Windows-compatible)
                latest_summary = self.output_dir / "latest_training_summary.txt"
                try:
                    if latest_summary.exists() or latest_summary.is_symlink():
                        latest_summary.unlink()
                    import shutil
                    shutil.copy2(summary_file, latest_summary)
                    print(f"Latest training summary copied to: {latest_summary}")
                except Exception as e:
                    print(f"Error creating the latest training summary copy: {e}")

            except Exception as e:
                print(f"Error saving the training summary file: {e}")

            # Generate training result visualization
            try:
                print("Collecting data to generate the confusion matrix and loss curve...")

                # 1. Prepare class names
                class_names = [mscoco_category2name[i] for i in sorted(mscoco_category2name.keys())]

                # 2. Prepare loss data
                train_losses = train_loss_history if train_loss_history else None
                val_losses = val_loss_history if val_loss_history else None

                # 3. Prepare confusion matrix data (y_true, y_pred)
                y_true = []
                y_pred = []

                # Call the visualization function
                self.visualize_training_results(
                    y_true=y_true,
                    y_pred=y_pred,
                    class_names=class_names,
                    train_losses=train_losses,
                    val_losses=val_losses
                )
                print("Training visualization pipeline completed")

            except Exception as viz_error:
                print(f"Calling visualize_training_results failed: {viz_error}")
                import traceback
                traceback.print_exc()

        # After training, use the best evaluator to generate visualization results
        if self.output_dir and dist.is_main_process():
            best_evaluator_path = self.output_dir / 'best_coco_evaluator.pth'
            if best_evaluator_path.exists():
                print("Loading best coco evaluator for final visualization.")
                best_coco_evaluator = torch.load(best_evaluator_path)

                # Get the final statistics if needed
                final_stats = {}
                if 'bbox' in best_coco_evaluator.iou_types:
                    final_stats['coco_eval_bbox'] = best_coco_evaluator.coco_eval['bbox'].stats.tolist()

                save_and_visualize_eval_results(best_coco_evaluator, final_stats, self.output_dir)
                print("Final visualization generated from the best model.")

    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                                              self.val_dataloader, base_ds, self.device, self.output_dir)

        # New: print the AP@0.50 and AP@0.50-0.95 for each class
        from src.data.coco.coco_dataset import mscoco_category2name, mscoco_label2category

        # Ensure printing only in the main process even in distributed mode
        if dist.is_main_process():
            if coco_evaluator is None:
                print("Warning: coco_evaluator is None!")
            elif "bbox" not in coco_evaluator.coco_eval:
                print("Warning: no 'bbox' key in coco_evaluator.coco_eval!")
                print(f"Available keys: {list(coco_evaluator.coco_eval.keys())}")
            else:
                try:
                    precisions = coco_evaluator.coco_eval["bbox"].eval['precision']  # [IoU, recall, cls, area, maxDets]
                    cat_ids = coco_evaluator.coco_eval["bbox"].params.catIds

                    print("\n===== Per-class AP values =====")
                    precisions = coco_evaluator.coco_eval["bbox"].eval['precision']
                    results_list = []

                    for idx, catId in enumerate(cat_ids):
                        # AP@0.50
                        ap50 = precisions[0, :, idx, 0, 2]
                        ap50 = ap50[ap50 > -1]
                        ap50_score = float(ap50.mean()) if ap50.size > 0 else 0.0

                        # AP@0.50-0.95
                        ap_all = precisions[:, :, idx, 0, 2]
                        ap_all = ap_all[ap_all > -1]
                        ap_all_score = float(ap_all.mean()) if ap_all.size > 0 else 0.0

                        name = mscoco_category2name.get(catId, str(catId))
                        print(f"Class {name:<15}: AP@0.50 = {ap50_score:.4f}, AP@0.50-0.95 = {ap_all_score:.4f}")
                        results_list.append((name, ap50_score, ap_all_score))

                    # Also write the results to a file
                    if self.output_dir:
                        with open(self.output_dir / "class_ap_latest.txt", "w") as f:
                            f.write(f"Epoch: {self.last_epoch}\n")
                            f.write(f"{'Class':<20} {'AP@0.50':<10} {'AP@0.50-0.95':<15}\n")
                            f.write("-" * 50 + "\n")
                            for name, ap50, ap_all in results_list:
                                f.write(f"{name:<20} {ap50:<10.4f} {ap_all:<15.4f}\n")

                except Exception as e:
                    print(f"Error printing AP: {e}")
                    import traceback
                    traceback.print_exc()

        # Compute the model's FLOPs and parameter count, without printing detailed layer info
        batch_size = 1
        input_shape = (batch_size, 3, 640, 640)
        flops, macs, params = calculate_flops(model=self.model,
                                              input_shape=input_shape,
                                              output_as_string=True,
                                              output_precision=4,
                                              print_detailed=False)  # Set to not print detailed information
        print("Model FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))

        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return

    def visualize_training_results(self, y_true, y_pred, class_names, train_losses=None, val_losses=None, error_stats=None):
        """
        Generate the confusion matrix, loss curve, and error analysis plots
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        import numpy as np

        viz_dir = self.output_dir / "training_visualizations"
        viz_dir.mkdir(exist_ok=True)

        # 1. Confusion matrix
        if y_true is not None and y_pred is not None and class_names:
            cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(viz_dir / "confusion_matrix.png", dpi=300)
            plt.close()
            print(f"Confusion matrix saved to: {viz_dir / 'confusion_matrix.png'}")
        else:
            print("Data required for the confusion matrix is missing; no confusion matrix generated.")

        # 2. Loss curve
        if train_losses is not None or val_losses is not None:
            plt.figure(figsize=(8, 6))
            if train_losses is not None:
                plt.plot(train_losses, label='Train Loss', color='blue')
            if val_losses is not None:
                plt.plot(val_losses, label='Val Loss', color='orange')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(viz_dir / "loss_curve.png", dpi=300)
            plt.close()
            print(f"Loss curve saved to: {viz_dir / 'loss_curve.png'}")
        else:
            print("No loss data provided; no loss curve generated.")

        # 3. Error analysis plot (e.g., bar chart of per-class false positives/misses)
        if error_stats is not None and isinstance(error_stats, dict):
            # error_stats: {'miss': [int,...], 'fp': [int,...], 'class_names': [str,...]}
            class_names = error_stats.get('class_names', class_names)
            miss = error_stats.get('miss', None)
            fp = error_stats.get('fp', None)
            if miss is not None or fp is not None:
                x = np.arange(len(class_names))
                plt.figure(figsize=(10, 6))
                if miss is not None:
                    plt.bar(x-0.2, miss, width=0.4, label='Missed', color='red')
                if fp is not None:
                    plt.bar(x+0.2, fp, width=0.4, label='False Positive', color='green')
                plt.xticks(x, class_names, rotation=45)
                plt.ylabel('Count')
                plt.title('Error Analysis by Class')
                plt.legend()
                plt.tight_layout()
                plt.savefig(viz_dir / "error_analysis.png", dpi=300)
                plt.close()
                print(f"Error analysis plot saved to: {viz_dir / 'error_analysis.png'}")
            else:
                print("error_stats does not contain miss/fp; no error analysis plot generated.")
        else:
            print("No error_stats provided; no error analysis plot generated.")