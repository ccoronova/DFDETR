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
matplotlib.use('Agg')  # 避免需要图形界面

import torch

from src.misc import dist
from src.data import get_coco_api_from_dataset
from src.data.coco.coco_dataset import mscoco_category2name, mscoco_label2category

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate, save_and_visualize_eval_results
from calflops import calculate_flops

# 添加自定义JSON编码器以处理NumPy类型
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
        
        # 跟踪最佳mAP@0.50值
        best_map50 = 0.0
        best_map50_epoch = -1
        
        # 存储最佳模型的详细指标信息
        best_map50_details = {}

        # 初始化 Loss 历史记录列表
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

            # 记录训练 Loss
            if 'loss' in train_stats:
                train_loss_history.append(train_stats['loss'])

            # Step the LR scheduler
            self.lr_scheduler.step()

            # 只保存最新的checkpoint，不再按步骤保存多个checkpoint
            if self.output_dir:
                checkpoint_path = self.output_dir / 'checkpoint.pth'
                dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            # Evaluate the model
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            # 记录验证 Loss
            if 'loss' in test_stats:
                val_loss = test_stats['loss']
                if isinstance(val_loss, list):
                     val_loss = val_loss[0]
                val_loss_history.append(val_loss)

            # Track IoU@0.50
            if coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
                iou_50 = coco_evaluator.coco_eval["bbox"].stats[1]  # 1 对应的是 IoU@0.50
                map50_95 = coco_evaluator.coco_eval["bbox"].stats[0]  # 0 对应的是 mAP@0.50-0.95
                
                print(f"Epoch {epoch}, IoU@0.50: {iou_50:.4f}, mAP@0.50-0.95: {map50_95:.4f}")

                # Update the best IoU@0.50 and corresponding epoch
                if iou_50 > best_stat['best_iou_50']:
                    best_stat['best_iou_50'] = iou_50
                    best_stat['epoch'] = epoch
                    print(f"New best IoU@0.50: {best_stat['best_iou_50']:.4f} at Epoch {best_stat['epoch']}")
                
                # 跟踪最佳mAP@0.50值
                if iou_50 > best_map50:
                    best_map50 = iou_50
                    best_map50_epoch = epoch
                    # 保存最佳mAP@0.50模型
                    if self.output_dir and dist.is_main_process():
                        best_map50_path = self.output_dir / 'best_map50.pth'
                        dist.save_on_master(self.state_dict(epoch), best_map50_path)
                        print(f"保存最佳mAP@0.50模型，指标: {best_map50:.4f}，轮次: {best_map50_epoch}")
                        
                        # 保存最佳 evaluator 用于后续可视化
                        best_evaluator_path = self.output_dir / 'best_coco_evaluator.pth'
                        torch.save(coco_evaluator, best_evaluator_path)

                        # 保存最佳模型的详细指标
                        best_map50_details = {
                            'epoch': epoch,
                            'iou_50': iou_50,
                            'map_50_95': map50_95,
                            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'other_metrics': {k: v for k, v in test_stats.items() if k not in ['loss', 'time']},
                            'class_metrics': {}
                        }
                        
                        # 添加每个类别的详细指标
                        try:
                            precisions = coco_evaluator.coco_eval["bbox"].eval['precision']
                            cat_ids = coco_evaluator.coco_eval["bbox"].params.catIds
                            
                            # 只保存需要的评估指标：AP@0.50-0.95（索引0）和AP@0.50（索引1）
                            all_stats = coco_evaluator.coco_eval["bbox"].stats.tolist()
                            # 只保存我们关心的两个指标
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
                                
                                # 获取类别名
                                label = mscoco_category2name.get(catId, str(catId))
                                
                                # 存储到详情中
                                best_map50_details['class_metrics'][label] = {
                                    'id': catId,
                                    'AP@0.50': ap50_score,
                                    'AP@0.50-0.95': ap_all_score
                                }
                        except Exception as e:
                            print(f"保存类别指标详情时出错: {e}")
                            # traceback.print_exc()
                        
                        # 确保目录存在
                        os.makedirs(self.output_dir, exist_ok=True)
                        
                        # 保存JSON格式
                        try:
                            json_path = self.output_dir / "best_map50_details.json"
                            with open(json_path, "w") as f:
                                json.dump(best_map50_details, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
                        except Exception as e:
                            print(f"保存JSON指标详情时出错: {e}")
                        
                        # 同时创建一个易于阅读的txt版本
                        try:
                            txt_path = self.output_dir / "best_map50_details.txt"
                            with open(txt_path, "w") as f:
                                f.write(f"===== 最佳mAP@0.50模型详情 =====\n\n")
                                f.write(f"训练轮次: {epoch}\n")
                                f.write(f"保存日期: {best_map50_details['date']}\n")
                                f.write(f"模型路径: {self.output_dir / 'best_map50.pth'}\n\n")
                                
                                f.write("===== 整体性能指标 =====\n")
                                f.write(f"mAP@0.50: {iou_50:.4f}\n\n")
                                
                                # 类别性能表格头
                                f.write("===== 各类别性能指标 =====\n")
                                f.write("| {:<20} | {:<10} |\n".format("Class Name", "AP@0.50"))
                                f.write("|" + "-"*22 + "|" + "-"*12 + "|\n")
                                
                                # 类别性能数据行
                                if best_map50_details['class_metrics']:
                                    for label, metrics in sorted(best_map50_details['class_metrics'].items()):
                                        if label and metrics and 'AP@0.50' in metrics:
                                            ap50 = metrics['AP@0.50']
                                            if not math.isnan(ap50):
                                                f.write("| {:<20} | {:<10.4f} |\n".format(
                                                    label[:20], ap50))
                                else:
                                    # 如果没有类别指标，添加提示信息
                                    f.write("| 暂无类别指标数据 - 模型可能还在早期训练阶段 |\n")
                            
                        except Exception as e:
                            print(f"保存TXT指标详情时出错: {e}")
                
                # 跟踪最佳mAP@0.50-0.95值
                # 已去除best_map50_95相关保存逻辑

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

                # for evaluation logs - 只保存最新的评估结果
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                  self.output_dir / "eval" / 'latest.pth')
                        
                        # 同时保存最佳mAP模型的评估结果
                        if iou_50 == best_map50:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                      self.output_dir / "eval" / 'best_map50.pth')

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        
        # 训练结束后，专门加载最佳模型并进行评估，以生成最佳模型的评估结果
        # 先评估最佳mAP@0.50模型
        if dist.is_main_process() and best_map50_epoch >= 0:
            print("\n" + "="*60)
            print("正在加载最佳mAP@0.50模型并重新评估，以生成最佳模型的可视化结果...")
            best_map50_path = self.output_dir / 'best_map50.pth'
            
            if best_map50_path.exists():
                # 保存当前模型状态
                current_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                
                try:
                    # 加载最佳mAP@0.50模型
                    best_state_dict = torch.load(best_map50_path, map_location=self.device)
                    
                    print(f"加载最佳mAP@0.50模型(Epoch {best_map50_epoch})...")
                    # 优先使用 EMA 分支进行再评估，确保与训练期评估一致
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
                    print(f"使用 {loaded_branch} 权重进行再评估，以保证与训练阶段的一致性。")
                    
                    # 创建专门的目录来存放最佳模型的评估结果
                    best_map50_eval_dir = self.output_dir / "best_map50_eval"
                    best_map50_eval_dir.mkdir(exist_ok=True)
                    
                    print(f"使用最佳mAP@0.50模型进行评估，结果将保存在 {best_map50_eval_dir}...")
                    
                    # 重新评估最佳模型
                    base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
                    test_stats, best_coco_evaluator = evaluate(
                        self.model, self.criterion, self.postprocessor, 
                        self.val_dataloader, base_ds, self.device, best_map50_eval_dir
                    )
                    # 打印对齐性：再评估的 mAP@0.50 与训练期记录的最佳值对比
                    try:
                        if best_coco_evaluator is not None and "bbox" in best_coco_evaluator.coco_eval:
                            reeval_map50 = float(best_coco_evaluator.coco_eval["bbox"].stats[1])
                            print(f"再评估 mAP@0.50: {reeval_map50:.4f}（训练期最佳: {best_map50:.4f}）")
                    except Exception:
                        pass
                    
                    # 保存评估结果，这个评估结果会由det_engine.py中的save_and_visualize_eval_results函数处理
                    # 生成对应的PR曲线和评估图表，并保存在best_map50_eval目录下
                    if best_coco_evaluator is not None and "bbox" in best_coco_evaluator.coco_eval:
                        torch.save(
                            best_coco_evaluator.coco_eval["bbox"].eval,
                            best_map50_eval_dir / "best_map50_eval_result.pth"
                        )
                        print(f"最佳mAP@0.50模型的评估结果已保存")
                        # 新增：保存PR曲线和ap_results.json
                        from src.solver.det_engine import save_and_visualize_eval_results
                        print(f"开始生成最佳模型的可视化结果...")
                        try:
                            save_and_visualize_eval_results(
                                best_coco_evaluator, test_stats, best_map50_eval_dir
                            )
                            print(f"✓ 最佳mAP@0.50模型的可视化结果已生成在: {best_map50_eval_dir / 'eval_visualizations'}")
                        except Exception as viz_error:
                            print(f"生成可视化时出错: {viz_error}")
                            import traceback
                            traceback.print_exc()
                        
                        # 复制生成的可视化文件到主可视化目录，以便统一查看
                        # vis_source_dir = best_map50_eval_dir / "eval_visualizations"
                        # vis_target_dir = self.output_dir / "visualizations" / "best_map50"
                        # if vis_source_dir.exists():
                        #     vis_target_dir.mkdir(exist_ok=True, parents=True)
                        #     import shutil
                        #     for vis_file in vis_source_dir.glob("*"):
                        #         if vis_file.is_file():
                        #             shutil.copy2(vis_file, vis_target_dir / vis_file.name)
                        #     print(f"最佳mAP@0.50模型的可视化结果已复制到 {vis_target_dir}")
                    
                    # 恢复模型到之前的状态
                    self.model.load_state_dict(current_model_state)
                    print("模型状态已恢复到训练结束时的状态")
                except Exception as e:
                    print(f"加载和评估最佳mAP@0.50模型时出错: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"警告: 最佳mAP@0.50模型文件不存在: {best_map50_path}")
        
        # 打印最佳模型指标总结
        print("\n" + "="*60)
        print("               训练结束，最佳模型总结               ")
        print("="*60)
        
        # 加载保存的最佳模型详细信息
        best_map50_file = self.output_dir / "best_map50_details.json"
        
        # 加载详细指标数据(如果存在)
        best_map50_data = {}
        
        if best_map50_file.exists() and dist.is_main_process():
            try:
                with open(best_map50_file, "r") as bf:
                    best_map50_data = json.load(bf)
            except Exception as e:
                print(f"读取最佳mAP@0.50指标文件时出错: {e}")
        
        # 打印最佳mAP@0.50模型信息
        print("\n【最佳mAP@0.50模型】")
        print("-"*60)
        print(f"指标值: {best_map50:.4f}")
        print(f"训练轮次: {best_map50_epoch}")
        print(f"模型路径: {self.output_dir / 'best_map50.pth'}")
        
        # 打印类别指标表格
        if "class_metrics" in best_map50_data and best_map50_data["class_metrics"]:
            print("\n各类别AP值:")
            print("{:<20} | {:<10}".format("Class Name", "AP@0.50"))
            print("-" * 33)
            
            for cls_name, metrics in sorted(best_map50_data["class_metrics"].items()):
                ap50 = metrics.get('AP@0.50', float('nan'))
                print("{:<20} | {:<10.4f}".format(cls_name[:20], ap50))
        
        print("\n" + "-"*60)
        print(f"最终checkpoint路径: {self.output_dir / 'checkpoint.pth'}")
        print("="*60)
        
        # 在训练结束时创建总结文件包含最佳模型的所有指标
        # 在训练结束时创建总结文件包含最佳模型的所有指标
        if dist.is_main_process() and self.output_dir:
            # 确保输出目录存在（self.output_dir 是 Path）
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # 使用时间戳创建唯一的总结文件名
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            summary_file = self.output_dir / f"training_summary_{timestamp}.txt"

            try:
                # 创建总结文件
                with open(summary_file, "w", encoding="utf-8") as f:
                    f.write("="*60 + "\n")
                    f.write("                  RT-DETR 训练总结                  \n")
                    f.write("="*60 + "\n\n")
                    
                    # 基本训练信息
                    f.write("【训练基本信息】\n")
                    f.write(f"- 完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"- 训练轮次: {args.epoches}\n")
                    f.write(f"- 训练时长: {total_time_str}\n")
                    f.write(f"- 模型参数: {n_parameters:,} 个参数\n")
                    
                    # 计算模型计算量
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
                    f.write(f"- 计算参数: {params}\n")
                    f.write(f"- 输出目录: {self.output_dir}\n\n")
                    
                    # 最佳mAP@0.50模型
                    f.write("-"*60 + "\n")
                    f.write("【最佳mAP@0.50模型】\n")
                    f.write("-"*60 + "\n")
                    f.write(f"- 指标值: {best_map50:.4f}\n")
                    f.write(f"- 训练轮次: {best_map50_epoch}\n")
                    f.write(f"- 模型路径: {self.output_dir / 'best_map50.pth'}\n")
                    if best_map50_data:
                        f.write(f"- 保存日期: {best_map50_data.get('date', '未记录')}\n")
                    
                    # 如果存在类别指标，添加表格形式的详细信息
                    if "class_metrics" in best_map50_data and best_map50_data["class_metrics"]:
                        f.write("\n【各类别AP值】\n")
                        f.write("| {:<20} | {:<10} |\n".format("类别名称", "AP@0.50"))
                        f.write("|" + "-"*22 + "|" + "-"*12 + "|\n")
                        
                        for cls_name, metrics in sorted(best_map50_data["class_metrics"].items()):
                            ap50 = metrics.get('AP@0.50', float('nan'))
                            f.write("| {:<20} | {:<10.4f} |\n".format(cls_name[:20], ap50))
                    f.write("\n\n" + "="*60 + "\n")
                    f.write("训练总结生成日期: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
                    f.write("="*60 + "\n")
                
                print(f"\n训练总结已保存到: {summary_file}")
                
                # 创建一个最新训练总结的拷贝（代替符号链接，更兼容 Windows）
                latest_summary = self.output_dir / "latest_training_summary.txt"
                try:
                    if latest_summary.exists() or latest_summary.is_symlink():
                        latest_summary.unlink()
                    import shutil
                    shutil.copy2(summary_file, latest_summary)
                    print(f"最新训练总结已复制到: {latest_summary}")
                except Exception as e:
                    print(f"创建最新训练总结拷贝时出错: {e}")
            
            except Exception as e:
                print(f"保存训练总结文件时出错: {e}")
            
            # 生成训练结果可视化
            try:
                print("正在收集数据以生成混淆矩阵和损失曲线...")
                
                # 1. 准备类别名称
                class_names = [mscoco_category2name[i] for i in sorted(mscoco_category2name.keys())]
                
                # 2. 准备 Loss 数据
                train_losses = train_loss_history if train_loss_history else None
                val_losses = val_loss_history if val_loss_history else None
                
                # 3. 准备混淆矩阵数据 (y_true, y_pred)
                y_true = [] 
                y_pred = []
                
                # 调用可视化函数
                self.visualize_training_results(
                    y_true=y_true, 
                    y_pred=y_pred, 
                    class_names=class_names,
                    train_losses=train_losses,
                    val_losses=val_losses
                )
                print("训练可视化流程执行完毕")
                
            except Exception as viz_error:
                print(f"调用 visualize_training_results 失败: {viz_error}")
                import traceback
                traceback.print_exc()

        # 在训练结束后，使用最佳评估器生成可视化结果
        if self.output_dir and dist.is_main_process():
            best_evaluator_path = self.output_dir / 'best_coco_evaluator.pth'
            if best_evaluator_path.exists():
                print("Loading best coco evaluator for final visualization.")
                best_coco_evaluator = torch.load(best_evaluator_path)
                
                # 获取最终的统计数据，如果需要的话
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

        # 新增：打印每个类别的AP@0.50和AP@0.50-0.95
        from src.data.coco.coco_dataset import mscoco_category2name, mscoco_label2category
        
        # 确保即使在分布式环境中也只在主进程中打印
        if dist.is_main_process():
            if coco_evaluator is None:
                print("警告: coco_evaluator 为 None！")
            elif "bbox" not in coco_evaluator.coco_eval:
                print("警告: coco_evaluator.coco_eval 中没有 'bbox' 键!")
                print(f"可用的键: {list(coco_evaluator.coco_eval.keys())}")
            else:
                try:
                    precisions = coco_evaluator.coco_eval["bbox"].eval['precision']  # [IoU, recall, cls, area, maxDets]
                    cat_ids = coco_evaluator.coco_eval["bbox"].params.catIds
                    
                    print("\n===== 每个类别的AP值 =====")
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
                        print(f"类别 {name:<15}: AP@0.50 = {ap50_score:.4f}, AP@0.50-0.95 = {ap_all_score:.4f}")
                        results_list.append((name, ap50_score, ap_all_score))
                        
                    # 将结果也写入到文件
                    if self.output_dir:
                        with open(self.output_dir / "class_ap_latest.txt", "w") as f:
                            f.write(f"Epoch: {self.last_epoch}\n")
                            f.write(f"{'Class':<20} {'AP@0.50':<10} {'AP@0.50-0.95':<15}\n")
                            f.write("-" * 50 + "\n")
                            for name, ap50, ap_all in results_list:
                                f.write(f"{name:<20} {ap50:<10.4f} {ap_all:<15.4f}\n")
                            
                except Exception as e:
                    print(f"打印AP时出错: {e}")
                    import traceback
                    traceback.print_exc()

        # 计算模型的计算量和参数量，但不打印详细的中间层信息
        batch_size = 1
        input_shape = (batch_size, 3, 640, 640)
        flops, macs, params = calculate_flops(model=self.model,
                                              input_shape=input_shape,
                                              output_as_string=True,
                                              output_precision=4,
                                              print_detailed=False)  # 设置不打印详细信息
        print("Model FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))

        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return

    def visualize_training_results(self, y_true, y_pred, class_names, train_losses=None, val_losses=None, error_stats=None):
        """
        生成混淆矩阵、损失曲线和错误分析图
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        import numpy as np

        viz_dir = self.output_dir / "training_visualizations"
        viz_dir.mkdir(exist_ok=True)

        # 1. 混淆矩阵
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
            print(f"✓ 混淆矩阵已保存到: {viz_dir / 'confusion_matrix.png'}")
        else:
            print("缺少混淆矩阵所需数据，未生成混淆矩阵。")

        # 2. 损失曲线
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
            print(f"✓ 损失曲线已保存到: {viz_dir / 'loss_curve.png'}")
        else:
            print("未提供损失数据，未生成损失曲线。")

        # 3. 错误分析图（如每类误检/漏检数量柱状图）
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
                print(f"✓ 错误分析图已保存到: {viz_dir / 'error_analysis.png'}")
            else:
                print("error_stats 未包含 miss/fp，未生成错误分析图。")
        else:
            print("未提供 error_stats，未生成错误分析图。")