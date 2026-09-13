import json
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
import pandas as pd
from collections import defaultdict
from pathlib import Path

matplotlib.use('Agg')  # Use non-interactive backend

# Set English fonts and styles for plots
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 设置输出目录
output_dir = "outputs/loss_analysis"
os.makedirs(output_dir, exist_ok=True)

# 读取txt文件中的数据
log_path = "outputs/log.txt"
with open(log_path, 'r') as file:
    lines = file.readlines()

# 初始化数据结构
epochs = []
metrics = defaultdict(list)

# 解析每一行的JSON数据
for line in lines:
    try:
        data = json.loads(line.strip())
        
        # 添加epoch
        if 'epoch' in data:
            epochs.append(data['epoch'])
            
        # 收集所有指标
        for key, value in data.items():
            if isinstance(value, (int, float)) or (isinstance(value, list) and all(isinstance(x, (int, float)) for x in value)):
                if isinstance(value, list):
                    # 处理列表值（比如精度和召回率）
                    metrics[key].append(value[0] if value else 0)
                else:
                    metrics[key].append(value)
    except json.JSONDecodeError:
        print(f"Unable to parse line: {line}")
    except Exception as e:
        print(f"Error processing line: {e}")

# 创建数据分析函数
def analyze_loss_components():
    """分析各损失组件的相对贡献并提供建议"""
    # Ensure sufficient data
    if len(epochs) == 0:
        return "没有足够的训练数据进行分析"
    
    # Calculate average loss from the last few epochs
    last_n = min(5, len(epochs))  # Take last 5 epochs or all if less than 5
    
    # 分离不同类型的损失
    main_losses = {'vfl': [], 'bbox': [], 'giou': []}
    aux_losses = {'vfl': [], 'bbox': [], 'giou': []}
    dn_losses = {'vfl': [], 'bbox': [], 'giou': []}
    
    # 收集最后n个epoch的数据
    for i in range(-last_n, 0):
        try:
            # 主损失
            if 'train_loss_vfl' in metrics and len(metrics['train_loss_vfl']) > abs(i):
                main_losses['vfl'].append(metrics['train_loss_vfl'][i])
            if 'train_loss_bbox' in metrics and len(metrics['train_loss_bbox']) > abs(i):
                main_losses['bbox'].append(metrics['train_loss_bbox'][i])
            if 'train_loss_giou' in metrics and len(metrics['train_loss_giou']) > abs(i):
                main_losses['giou'].append(metrics['train_loss_giou'][i])
            
            # aux损失（取平均）
            aux_vfl = 0
            aux_bbox = 0
            aux_giou = 0
            aux_count = 0
            
            for j in range(3):  # 假设有3个aux损失
                aux_key_vfl = f'train_loss_vfl_aux_{j}'
                aux_key_bbox = f'train_loss_bbox_aux_{j}'
                aux_key_giou = f'train_loss_giou_aux_{j}'
                
                if aux_key_vfl in metrics and len(metrics[aux_key_vfl]) > abs(i):
                    aux_vfl += metrics[aux_key_vfl][i]
                    aux_count += 1
                if aux_key_bbox in metrics and len(metrics[aux_key_bbox]) > abs(i):
                    aux_bbox += metrics[aux_key_bbox][i]
                if aux_key_giou in metrics and len(metrics[aux_key_giou]) > abs(i):
                    aux_giou += metrics[aux_key_giou][i]
            
            if aux_count > 0:
                aux_losses['vfl'].append(aux_vfl / aux_count)
                aux_losses['bbox'].append(aux_bbox / aux_count)
                aux_losses['giou'].append(aux_giou / aux_count)
            
            # dn损失（取平均）
            dn_vfl = 0
            dn_bbox = 0
            dn_giou = 0
            dn_count = 0
            
            for j in range(3):  # 假设有3个dn损失
                dn_key_vfl = f'train_loss_vfl_dn_{j}'
                dn_key_bbox = f'train_loss_bbox_dn_{j}'
                dn_key_giou = f'train_loss_giou_dn_{j}'
                
                if dn_key_vfl in metrics and len(metrics[dn_key_vfl]) > abs(i):
                    dn_vfl += metrics[dn_key_vfl][i]
                    dn_count += 1
                if dn_key_bbox in metrics and len(metrics[dn_key_bbox]) > abs(i):
                    dn_bbox += metrics[dn_key_bbox][i]
                if dn_key_giou in metrics and len(metrics[dn_key_giou]) > abs(i):
                    dn_giou += metrics[dn_key_giou][i]
            
            if dn_count > 0:
                dn_losses['vfl'].append(dn_vfl / dn_count)
                dn_losses['bbox'].append(dn_bbox / dn_count)
                dn_losses['giou'].append(dn_giou / dn_count)
                
        except Exception as e:
            print(f"分析过程中出错: {e}")
    
    # 计算平均值
    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0
    
    avg_main = {k: safe_mean(v) for k, v in main_losses.items()}
    avg_aux = {k: safe_mean(v) for k, v in aux_losses.items()}
    avg_dn = {k: safe_mean(v) for k, v in dn_losses.items()}
    
    # 计算主要损失比例
    total_main = sum(avg_main.values())
    loss_ratios = {k: v/total_main if total_main else 0 for k, v in avg_main.items()}
    
    # 建议计算
    suggestions = []
    
    # 1. Analyze ratio between vfl, bbox and giou losses
    if loss_ratios['giou'] > 0.7:
        suggestions.append("GIoU loss ratio is too high (> 70%), consider decreasing giou_loss_coef")
    elif loss_ratios['giou'] < 0.3:
        suggestions.append("GIoU loss ratio is too low (< 30%), consider increasing giou_loss_coef")
    
    if loss_ratios['bbox'] > 0.4:
        suggestions.append("L1 bbox loss ratio is too high (> 40%), consider decreasing bbox_loss_coef")
    elif loss_ratios['bbox'] < 0.1:
        suggestions.append("L1 bbox loss ratio is too low (< 10%), consider increasing bbox_loss_coef")
        
    if avg_main['vfl'] > 1.0:
        suggestions.append("VFL classification loss is high (>1.0), consider increasing class balance or decreasing vfl_loss_coef")
    
    # 2. Analyze auxiliary and DN losses
    if safe_mean(aux_losses['giou']) > safe_mean(main_losses['giou']) * 1.2:
        suggestions.append("Auxiliary GIoU loss is significantly higher than main GIoU loss, consider decreasing aux_loss_coef")
    
    if safe_mean(dn_losses['vfl']) > safe_mean(main_losses['vfl']) * 1.5:
        suggestions.append("DN classification loss is significantly higher than main classification loss, consider decreasing dn_weight or increasing dn samples")
    
    if safe_mean(dn_losses['bbox']) < safe_mean(main_losses['bbox']) * 0.5:
        suggestions.append("DN bbox loss is significantly lower than main bbox loss, DN might not be effective, consider increasing dn_weight")
    
    # 3. Overall loss parameter recommendations
    main_total = total_main if total_main else 1
    aux_total = sum(avg_aux.values())
    dn_total = sum(avg_dn.values())
    
    if dn_total < main_total * 0.4 and 'train_loss_vfl_dn_0' in metrics:
        suggestions.append("DN total loss is too small, consider increasing dn_weight (currently might be too low)")
    elif dn_total > main_total * 2 and 'train_loss_vfl_dn_0' in metrics:
        suggestions.append("DN total loss is too high, consider decreasing dn_weight (currently might be too high)")
    
    if aux_total < main_total * 0.4 and 'train_loss_vfl_aux_0' in metrics:
        suggestions.append("Auxiliary total loss is too small, consider increasing aux_loss_coef")
    elif aux_total > main_total * 2 and 'train_loss_vfl_aux_0' in metrics:
        suggestions.append("Auxiliary total loss is too high, consider decreasing aux_loss_coef")
    
    # Recommend loss function weights based on loss ratios
    if loss_ratios:
        rec_giou = round(1.0 if loss_ratios['giou'] < 0.5 else (0.5 if loss_ratios['giou'] > 0.7 else 0.8), 1)
        rec_bbox = round(5.0 if loss_ratios['bbox'] < 0.2 else (2.0 if loss_ratios['bbox'] > 0.4 else 3.0), 1)
        rec_cls = round(1.0, 1)  # VFL usually kept at 1.0
        
        suggestions.append(f"Recommended loss function weights: giou_loss_coef={rec_giou}, bbox_loss_coef={rec_bbox}, vfl_loss_coef={rec_cls}")
    
    # 4. Check precision and recall
    if 'test_precision@0.5' in metrics and 'test_recall@0.5' in metrics:
        avg_precision = safe_mean(metrics['test_precision@0.5'][-last_n:])
        avg_recall = safe_mean(metrics['test_recall@0.5'][-last_n:])
        
        if avg_precision < 0.1:
            suggestions.append(f"Test precision is very low ({avg_precision:.4f}), check class imbalance issues and classification loss weights")
        
        if avg_recall < 0.1:
            suggestions.append(f"Test recall is very low ({avg_recall:.4f}), consider increasing GIoU loss weight to improve localization")
    
    # 格式化建议输出
    result = "损失函数参数分析与建议:\n" + "-"*50 + "\n"
    result += f"主要损失平均值: VFL={avg_main['vfl']:.4f}, BBOX={avg_main['bbox']:.4f}, GIOU={avg_main['giou']:.4f}\n"
    result += f"损失比例: VFL={loss_ratios['vfl']*100:.1f}%, BBOX={loss_ratios['bbox']*100:.1f}%, GIOU={loss_ratios['giou']*100:.1f}%\n"
    
    if avg_aux['vfl'] or avg_aux['bbox'] or avg_aux['giou']:
        result += f"辅助损失平均值: VFL={avg_aux['vfl']:.4f}, BBOX={avg_aux['bbox']:.4f}, GIOU={avg_aux['giou']:.4f}\n"
    
    if avg_dn['vfl'] or avg_dn['bbox'] or avg_dn['giou']:
        result += f"DN损失平均值: VFL={avg_dn['vfl']:.4f}, BBOX={avg_dn['bbox']:.4f}, GIOU={avg_dn['giou']:.4f}\n"
    
    result += "\n建议:\n"
    for i, sugg in enumerate(suggestions, 1):
        result += f"{i}. {sugg}\n"
    
    return result

# Plotting function
def plot_detailed_losses():
    """Draw detailed loss curves"""
    if not epochs:
        print("Not enough data for plotting")
        return
    
    # 1. Total loss curve
    plt.figure(figsize=(12, 8))
    if 'train_loss' in metrics:
        plt.plot(epochs, metrics['train_loss'], label='Total Loss', marker='o', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Training Total Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'total_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Main loss components comparison
    plt.figure(figsize=(12, 8))
    components = [
        ('train_loss_vfl', 'Classification Loss (VFL)'),
        ('train_loss_bbox', 'Bounding Box Loss (L1)'),
        ('train_loss_giou', 'IoU Loss (GIoU)')
    ]
    for key, label in components:
        if key in metrics:
            plt.plot(epochs, metrics[key], label=label, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Main Loss Components Comparison')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'main_loss_components.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Auxiliary loss components
    plt.figure(figsize=(12, 8))
    
    # Collect all auxiliary loss components
    aux_components = []
    for key in metrics.keys():
        if '_aux_' in key and len(metrics[key]) == len(epochs):
            aux_components.append((key, key.replace('train_loss_', '').replace('_', ' ').title()))
    
    for key, label in aux_components:
        plt.plot(epochs, metrics[key], label=label, marker='o')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Auxiliary Loss Components')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'aux_loss_components.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. DN loss components
    plt.figure(figsize=(12, 8))
    
    # Collect all DN loss components
    dn_components = []
    for key in metrics.keys():
        if '_dn_' in key and len(metrics[key]) == len(epochs):
            dn_components.append((key, key.replace('train_loss_', '').replace('_', ' ').title()))
    
    for key, label in dn_components:
        plt.plot(epochs, metrics[key], label=label, marker='o')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('DN Loss Components')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'dn_loss_components.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Precision and recall
    if 'test_precision@0.5' in metrics and 'test_recall@0.5' in metrics:
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, metrics['test_precision@0.5'], label='Precision@0.5', marker='o', color='blue')
        plt.plot(epochs, metrics['test_recall@0.5'], label='Recall@0.5', marker='s', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Test Precision and Recall')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'precision_recall.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Loss ratio stacked plot
    try:
        plt.figure(figsize=(12, 8))
        components = ['train_loss_vfl', 'train_loss_bbox', 'train_loss_giou']
        labels = ['Classification Loss (VFL)', 'Bounding Box Loss (L1)', 'IoU Loss (GIoU)']
        
        # Ensure all components have data
        if all(comp in metrics for comp in components) and all(len(metrics[comp]) == len(epochs) for comp in components):
            data = np.array([metrics[comp] for comp in components])
            plt.stackplot(epochs, data, labels=labels, alpha=0.8)
            plt.xlabel('Epoch')
            plt.ylabel('Loss Contribution')
            plt.title('Loss Components Contribution Over Time')
            plt.grid(True)
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(output_dir, 'loss_contribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating stacked plot: {e}")
    
    # 7. Learning rate changes
    if 'train_lr' in metrics:
        plt.figure(figsize=(12, 5))
        plt.plot(epochs, metrics['train_lr'], label='Learning Rate', marker='o', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Changes')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
        plt.close()

# 运行分析和可视化
plot_detailed_losses()
analysis_result = analyze_loss_components()

# 保存分析结果到文本文件
with open(os.path.join(output_dir, 'loss_analysis.txt'), 'w', encoding='utf-8') as f:
    f.write(analysis_result)

print(f"详细损失分析已保存至: {output_dir}")
print("-" * 50)
print(analysis_result)

# 学习率和epoch分析
if 'train_lr' in metrics and epochs:
    initial_lr = metrics['train_lr'][0] if metrics['train_lr'] else 0
    final_lr = metrics['train_lr'][-1] if metrics['train_lr'] else 0
    max_epoch = max(epochs) if epochs else 0
    print("\n学习率与训练轮数分析:")
    print("-" * 50)
    print(f"当前设置: 学习率={initial_lr:.8f}, 轮数={max_epoch}")
    # Learning rate recommendations
    if initial_lr < 1e-6:
        print("学习率可能过低，建议提升到 1e-5 ~ 5e-5")
    elif initial_lr > 1e-4:
        print("学习率可能过高，建议降低到 1e-5 ~ 5e-5")
    else:
        print(f"当前学习率 {initial_lr:.8f} 合理")
    # Epoch recommendations
    if 'train_loss' in metrics and len(metrics['train_loss']) >= 2:
        recent_loss_change = abs(metrics['train_loss'][-1] - metrics['train_loss'][-2])
        if max_epoch < 100:
            print(f"当前轮数设置（{max_epoch}）可能不足以收敛，建议增加到200-300轮")
        elif recent_loss_change > 0.01 * metrics['train_loss'][-1]:
            print(f"损失仍在显著变化（变化量: {recent_loss_change:.4f}），建议增加训练轮数")
        else:
            print(f"当前轮数设置（{max_epoch}）较为合适")
    # Final recommendation
    print("\n推荐设置:")
    recommended_lr = 2e-5 if initial_lr < 1e-6 or initial_lr > 1e-4 else initial_lr
    recommended_epochs = max(200, max_epoch + 50) if max_epoch < 150 else max_epoch
    print(f"学习率: {recommended_lr:.8f}")
    print(f"轮数: {recommended_epochs}")