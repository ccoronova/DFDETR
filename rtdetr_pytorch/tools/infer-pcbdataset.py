import torch
import torch.nn as nn
import torchvision.transforms as T
from sympy.physics.vector import outer
from torch.cuda.amp import autocast
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS
import numpy as np


def postprocess(labels, boxes, scores, iou_threshold=0.7):
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou

    merged_labels = []
    merged_boxes = []
    merged_scores = []
    used_indices = set()

    for i in range(len(boxes)):
        if i in used_indices:
            continue
        current_box = boxes[i]
        current_label = labels[i]
        current_score = scores[i]

        # # 强制将标签0替换为1 (或其他有效标签)
        # if current_label == 0:
        #     current_label = 1  # 或者你可以将标签0直接忽略掉

        boxes_to_merge = [current_box]
        scores_to_merge = [current_score]
        used_indices.add(i)

        for j in range(i + 1, len(boxes)):
            if j in used_indices:
                continue
            if labels[j] != current_label:
                continue
            other_box = boxes[j]
            iou = calculate_iou(current_box, other_box)
            if iou >= iou_threshold:
                boxes_to_merge.append(other_box.tolist())
                scores_to_merge.append(scores[j])
                used_indices.add(j)

        xs = np.concatenate([[box[0], box[2]] for box in boxes_to_merge])
        ys = np.concatenate([[box[1], box[3]] for box in boxes_to_merge])
        merged_box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
        merged_score = max(scores_to_merge)
        merged_boxes.append(merged_box)
        merged_labels.append(current_label)
        merged_scores.append(merged_score)

    return [np.array(merged_labels)], [np.array(merged_boxes)], [np.array(merged_scores)]



def slice_image(image, slice_height, slice_width, overlap_ratio):
    img_width, img_height = image.size

    slices = []
    coordinates = []
    step_x = int(slice_width * (1 - overlap_ratio))
    step_y = int(slice_height * (1 - overlap_ratio))

    for y in range(0, img_height, step_y):
        for x in range(0, img_width, step_x):
            box = (x, y, min(x + slice_width, img_width), min(y + slice_height, img_height))
            slice_img = image.crop(box)
            slices.append(slice_img)
            coordinates.append((x, y))
    return slices, coordinates
def merge_predictions(predictions, slice_coordinates, orig_image_size, slice_width, slice_height, threshold=0.30):
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    orig_height, orig_width = orig_image_size
    for i, (label, boxes, scores) in enumerate(predictions):
        x_shift, y_shift = slice_coordinates[i]
        scores = np.array(scores).reshape(-1)
        valid_indices = scores > threshold
        valid_labels = np.array(label).reshape(-1)[valid_indices]
        valid_boxes = np.array(boxes).reshape(-1, 4)[valid_indices]
        valid_scores = scores[valid_indices]
        for j, box in enumerate(valid_boxes):
            box[0] = np.clip(box[0] + x_shift, 0, orig_width)
            box[1] = np.clip(box[1] + y_shift, 0, orig_height)
            box[2] = np.clip(box[2] + x_shift, 0, orig_width)
            box[3] = np.clip(box[3] + y_shift, 0, orig_height)
            valid_boxes[j] = box
        merged_labels.extend(valid_labels)
        merged_boxes.extend(valid_boxes)
        merged_scores.extend(valid_scores)
    return np.array(merged_labels), np.array(merged_boxes), np.array(merged_scores)


def draw(images, labels_list, boxes_list, scores_list, thrh=0.7,
         path="outputs/infer/pcb-dataset/result", image_names=[]):
    os.makedirs(path, exist_ok=True)
    log_file_path = os.path.join(path, "log.txt")

    # 定义颜色映射（6种缺陷对应颜色）
    color_mapping = {
        0: "#FF0000",  # Missing Hole open - 红色
        1: "#00FF00",  # Mouse Bite short- 绿色
        2: "#0000FF",  # Open Circuit  mousebite - 蓝色
        3: "#FFFF00",  # Short -  spur  黄色
        4: "#FF00FF",  # Spur -   copper 品红
        5: "#00FFFF"  # Spurious Copper -   pin-hole 青色
    }
    # pcbdataset
    label_mapping = {
        0: 'missing_hole',
        1: 'mouse_bite',
        2: 'open_circuit',
        3: 'short',
        4: 'spur',
        5: 'spurious_copper',
    }
    # deeppcb
    # label_mapping = {
    #     0: 'open',
    #     1: 'short',
    #     2: 'mousebite',
    #     3: 'spur',
    #     4: 'copper',
    #     5: 'pin-hole'
    #     }
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        for i, im in enumerate(images):
            draw_obj = ImageDraw.Draw(im)
            scr = scores_list[i].detach().cpu().numpy()
            lab = labels_list[i].detach().cpu().numpy()
            box = boxes_list[i].detach().cpu().numpy()

            # 过滤结果
            filtered_indices = (scr > thrh)
            filtered_labels = lab[filtered_indices]
            filtered_boxes = box[filtered_indices]
            filtered_scores = scr[filtered_indices]
            # 写入日志文件（修复日志空置问题）
            log_file.write(f"图像 {image_names[i]} 的检测结果：\n")
            if len(filtered_labels) == 0:
                log_file.write("  无有效检测结果\n")
            else:
                for lbl, scr, bbox in zip(filtered_labels, filtered_scores, filtered_boxes):
                    log_file.write(f"  标签: {label_mapping[lbl]} 分数: {scr:.2f} 位置: {bbox.tolist()}\n")

            # 记录已绘制的文本区域
            drawn_text_areas = []

            im_width, im_height = im.size

            for j, b in enumerate(filtered_boxes):
                label_id = filtered_labels[j].item()
                label_name = label_mapping.get(label_id, 'Unknown')
                color = color_mapping.get(label_id, "#FF0000")

                # 设置字体
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 80)
                except:
                    try:
                        font = ImageFont.truetype("C:\\Windows\\Fonts\\simhei.ttf", 80)
                    except:
                        font = ImageFont.load_default()

                # 创建文本
                text = f"{label_name} {filtered_scores[j]:.2f}"
                text_bbox = draw_obj.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                # 添加padding（关键改进）
                padding = 5
                text_width += padding * 2
                text_height += padding * 2

                # 智能位置选择
                box_left, box_top, box_right, box_bottom = b
                box_center_x = (box_left + box_right) / 2
                box_center_y = (box_top + box_bottom) / 2

                # 候选位置：上、下、左、右
                candidate_positions = [
                    (box_left, box_top - text_height - 10),  # 上方
                    (box_left, box_bottom + 10),            # 下方
                    (box_left - text_width - 10, box_top),   # 左侧
                    (box_right + 10, box_top)                # 右侧
                ]

                best_pos = None
                min_overlap = float('inf')

                for pos in candidate_positions:
                    text_x, text_y = pos
                    text_x = np.clip(text_x, 0, im_width - text_width)
                    text_y = np.clip(text_y, 0, im_height - text_height)
                    current_rect = (text_x, text_y, text_x + text_width, text_y + text_height)

                    # 计算与已有文本区域的重叠
                    overlap = 0
                    for prev_rect in drawn_text_areas:
                        x_overlap = max(0, min(current_rect[2], prev_rect[2]) - max(current_rect[0], prev_rect[0]))
                        y_overlap = max(0, min(current_rect[3], prev_rect[3]) - max(current_rect[1], prev_rect[1]))
                        overlap += x_overlap * y_overlap

                    if overlap < min_overlap:
                        min_overlap = overlap
                        best_pos = (text_x, text_y)

                # 如果所有候选位置都有重叠，则选择重叠最小的位置
                if best_pos is None:
                    # 如果没有合适的位置，选择默认上方位置
                    best_pos = (box_left, box_top - text_height - 10)

                # 最终边界保护
                text_x, text_y = best_pos
                text_x = np.clip(text_x, 0, im_width - text_width)
                text_y = np.clip(text_y, 0, im_height - text_height)

                # 防止下方越界
                if text_y + text_height > im_height:
                    text_y = im_height - text_height

                # 记录文本区域
                drawn_text_areas.append((text_x, text_y, text_x + text_width, text_y + text_height))

                # 绘制背景
                draw_obj.rectangle(
                    [text_x, text_y, text_x + text_width, text_y + text_height],
                    fill=color + "80"  # 半透明背景
                )

                # 绘制文本
                draw_obj.text(
                    (text_x + padding, text_y + padding),
                    text,
                    font=font,
                    fill="black"
                )

                # 绘制边界框
                # draw_obj.rectangle(list(b), outline="#FF0000", width=8)
                draw_obj.rectangle(list(b), outline=color, width=8)
            # 保存图像
            im.save(os.path.join(path, f'results_{image_names[i]}'))
def load_model(cfg, checkpoint_path):
    """加载模型并恢复训练状态"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'ema' in checkpoint:
        state_dict = checkpoint['ema']['module']
    else:
        state_dict = checkpoint['model']

    try:
        # 加载模型时使用 strict=False 来忽略不匹配的参数
        cfg.model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        raise

    return cfg.model

def main(args, ):
    """main
    """
    # # 保存标准输出的原始引用
    # original_stdout = sys.stdout
    # # 打开文件以写入日志
    # with open('training_log.txt', 'w') as f:
    #     sys.stdout = f  # 将标准输出重定向到文件

    # 这里放置模型加载、训练等逻辑
    cfg = YAMLConfig(args.config, resume=args.resume)
    # 加载模型并恢复状态
    model = load_model(cfg, args.resume)
    model.to(args.device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    # print("开始训练...")  # 示例输出
    # for epoch in range(num_epochs):  # 假设有一个 num_epochs 变量
    #     print(f"Epoch {epoch + 1}/{num_epochs}")
    #     # 你的训练逻辑

    # sys.stdout = original_stdout  # 恢复标准输出
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)
    # im_pil = Image.open(args.im_file).convert('RGB')       #打开单个图片路径，convert为转化为rgb形式
    # w, h = im_pil.size                              #获取图片尺寸，宽和高
    # orig_size = torch.tensor([w, h])[None].to(args.device)      #创建一个pytorch张量（tensor），包含图片宽度和高度的一维数组

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
     # 用于存储处理的图像、标签、框、分数及其文件名
    images = []
    labels_list = []
    boxes_list = []
    scores_list = []
    image_names = []  # 在这里定义 image_names
    # Batch process all images in the specified directory,#遍历文件夹读取图片
    image_names = []  # 用于存储图像的文件名
    for image_name in os.listdir(args.im_dir):
        im_path = os.path.join(args.im_dir, image_name)
        if not im_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
    # 读取图片并转化为rgb
        im_pil = Image.open(im_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)
        im_data = transforms(im_pil)[None].to(args.device)
    # 前向传播获取检测结果
        output = model(im_data, orig_size)
        labels, boxes, scores = output
        # 将结果存储在列表中
        images.append(im_pil)
        labels_list.append(labels)
        boxes_list.append(boxes)
        scores_list.append(scores)

        image_names.append(image_name)  # 保存文件名
        # 在处理完所有图像后调用绘图函数
        print("Drawing results...")  # 添加调试打印
        draw(images, labels_list, boxes_list, scores_list, 0.46, path=args.output_dir,image_names=image_names)
        print("Results saved.")  # 添加调试打印



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,default=r'configs/rtdetr/rtdetr_r18vd_6x_coco.yml' )       #指定配置文件路径
    parser.add_argument('-r', '--resume', type=str, default=r'outputs/pcbdataset/best_map50.pth')   #指定训练权重文件路径
    # parser.add_argument('-f', '--im-file', type=str,default=r'path/to/image.jpg')                  #用于单张图片路径
    # parser.add_argument('-s', '--sliced', type=bool, default=False)                                                                                         #是否切片处理
    parser.add_argument('-d', '--device', type=str, default='cuda')                                                                                             #设备选择，cpu，cuda
    parser.add_argument('--im-dir', type=str, default=r'outputs/infer/pcb-dataset', help="Directory containing images to predict")
    # parser.add_argument('-nc', '--numberofboxes', type=int, default=25)                                                                                  #切片模式下的切片数量
    parser.add_argument('-o', '--output-dir', type=str, default=r'outputs/infer/pcb-dataset/result', help="Path to the directory where output images will be saved.")                                # 新增输出目录参数
    args = parser.parse_args()
    main(args)


