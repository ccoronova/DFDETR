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

        # # Forcefully replace label 0 with 1 (or another valid label)
        # if current_label == 0:
        #     current_label = 1  # or you can simply ignore label 0

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

    # Define the color mapping (colors for the 6 defect types)
    color_mapping = {
        0: "#FF0000",  # Missing Hole open - red
        1: "#00FF00",  # Mouse Bite short- green
        2: "#0000FF",  # Open Circuit  mousebite - blue
        3: "#FFFF00",  # Short -  spur  yellow
        4: "#FF00FF",  # Spur -   copper  magenta
        5: "#00FFFF"  # Spurious Copper -   pin-hole cyan
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

            # Filter results
            filtered_indices = (scr > thrh)
            filtered_labels = lab[filtered_indices]
            filtered_boxes = box[filtered_indices]
            filtered_scores = scr[filtered_indices]
            # Write to the log file (fixed the previously empty log)
            log_file.write(f"Detection results for image {image_names[i]}:\n")
            if len(filtered_labels) == 0:
                log_file.write("  No valid detections\n")
            else:
                for lbl, scr, bbox in zip(filtered_labels, filtered_scores, filtered_boxes):
                    log_file.write(f"  Label: {label_mapping[lbl]} Score: {scr:.2f} Location: {bbox.tolist()}\n")

            # Track already drawn text regions
            drawn_text_areas = []

            im_width, im_height = im.size

            for j, b in enumerate(filtered_boxes):
                label_id = filtered_labels[j].item()
                label_name = label_mapping.get(label_id, 'Unknown')
                color = color_mapping.get(label_id, "#FF0000")

                # Set the font
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 80)
                except:
                    try:
                        font = ImageFont.truetype("C:\\Windows\\Fonts\\simhei.ttf", 80)
                    except:
                        font = ImageFont.load_default()

                # Create the text
                text = f"{label_name} {filtered_scores[j]:.2f}"
                text_bbox = draw_obj.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                # Add padding (key improvement)
                padding = 5
                text_width += padding * 2
                text_height += padding * 2

                # Smart position selection
                box_left, box_top, box_right, box_bottom = b
                box_center_x = (box_left + box_right) / 2
                box_center_y = (box_top + box_bottom) / 2

                # Candidate positions: top, bottom, left, right
                candidate_positions = [
                    (box_left, box_top - text_height - 10),  # above
                    (box_left, box_bottom + 10),            # below
                    (box_left - text_width - 10, box_top),   # left
                    (box_right + 10, box_top)                # right
                ]

                best_pos = None
                min_overlap = float('inf')

                for pos in candidate_positions:
                    text_x, text_y = pos
                    text_x = np.clip(text_x, 0, im_width - text_width)
                    text_y = np.clip(text_y, 0, im_height - text_height)
                    current_rect = (text_x, text_y, text_x + text_width, text_y + text_height)

                    # Compute overlap with already drawn text regions
                    overlap = 0
                    for prev_rect in drawn_text_areas:
                        x_overlap = max(0, min(current_rect[2], prev_rect[2]) - max(current_rect[0], prev_rect[0]))
                        y_overlap = max(0, min(current_rect[3], prev_rect[3]) - max(current_rect[1], prev_rect[1]))
                        overlap += x_overlap * y_overlap

                    if overlap < min_overlap:
                        min_overlap = overlap
                        best_pos = (text_x, text_y)

                # If all candidate positions overlap, choose the one with the least overlap
                if best_pos is None:
                    # If no suitable position, choose the default position above
                    best_pos = (box_left, box_top - text_height - 10)

                # Final boundary protection
                text_x, text_y = best_pos
                text_x = np.clip(text_x, 0, im_width - text_width)
                text_y = np.clip(text_y, 0, im_height - text_height)

                # Prevent overflow below
                if text_y + text_height > im_height:
                    text_y = im_height - text_height

                # Record the text region
                drawn_text_areas.append((text_x, text_y, text_x + text_width, text_y + text_height))

                # Draw the background
                draw_obj.rectangle(
                    [text_x, text_y, text_x + text_width, text_y + text_height],
                    fill=color + "80"  # semi-transparent background
                )

                # Draw the text
                draw_obj.text(
                    (text_x + padding, text_y + padding),
                    text,
                    font=font,
                    fill="black"
                )

                # Draw the bounding box
                # draw_obj.rectangle(list(b), outline="#FF0000", width=8)
                draw_obj.rectangle(list(b), outline=color, width=8)
            # Save the image
            im.save(os.path.join(path, f'results_{image_names[i]}'))
def load_model(cfg, checkpoint_path):
    """Load the model and restore training state"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'ema' in checkpoint:
        state_dict = checkpoint['ema']['module']
    else:
        state_dict = checkpoint['model']

    try:
        # When loading the model, use strict=False to ignore mismatched parameters
        cfg.model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        raise

    return cfg.model

def main(args, ):
    """main
    """
    # # Save the original reference to standard output
    # original_stdout = sys.stdout
    # # Open a file to write the log
    # with open('training_log.txt', 'w') as f:
    #     sys.stdout = f  # redirect standard output to the file

    # Model loading, training, etc. logic goes here
    cfg = YAMLConfig(args.config, resume=args.resume)
    # Load the model and restore its state
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

    # print("Start training...")  # sample output
    # for epoch in range(num_epochs):  # assuming there is a num_epochs variable
    #     print(f"Epoch {epoch + 1}/{num_epochs}")
    #     # your training logic

    # sys.stdout = original_stdout  # restore standard output
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
    # im_pil = Image.open(args.im_file).convert('RGB')       # open a single image path; convert turns it into RGB
    # w, h = im_pil.size                              # get the image size, width and height
    # orig_size = torch.tensor([w, h])[None].to(args.device)      # create a PyTorch tensor with a 1-D array of image width and height

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
     # Used to store the processed images, labels, boxes, scores, and their file names
    images = []
    labels_list = []
    boxes_list = []
    scores_list = []
    image_names = []  # define image_names here
    # Batch process all images in the specified directory, # read images by iterating the folder
    image_names = []  # used to store image file names
    for image_name in os.listdir(args.im_dir):
        im_path = os.path.join(args.im_dir, image_name)
        if not im_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
    # Read the image and convert to RGB
        im_pil = Image.open(im_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)
        im_data = transforms(im_pil)[None].to(args.device)
    # Forward pass to obtain detection results
        output = model(im_data, orig_size)
        labels, boxes, scores = output
        # Store the results in lists
        images.append(im_pil)
        labels_list.append(labels)
        boxes_list.append(boxes)
        scores_list.append(scores)

        image_names.append(image_name)  # save the file name
        # Call the drawing function after all images have been processed
        print("Drawing results...")  # add debug print
        draw(images, labels_list, boxes_list, scores_list, 0.46, path=args.output_dir,image_names=image_names)
        print("Results saved.")  # add debug print



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,default=r'configs/rtdetr/rtdetr_r18vd_6x_coco.yml' )       #specify the path to the config file
    parser.add_argument('-r', '--resume', type=str, default=r'outputs/pcbdataset/best_map50.pth')   #specify the path to the training weight file
    # parser.add_argument('-f', '--im-file', type=str,default=r'path/to/image.jpg')                  #for a single image path
    # parser.add_argument('-s', '--sliced', type=bool, default=False)                                                                                         #whether to process slices
    parser.add_argument('-d', '--device', type=str, default='cuda')                                                                                             #device selection, cpu, cuda
    parser.add_argument('--im-dir', type=str, default=r'outputs/infer/pcb-dataset', help="Directory containing images to predict")
    # parser.add_argument('-nc', '--numberofboxes', type=int, default=25)                                                                                  #number of slices in slice mode
    parser.add_argument('-o', '--output-dir', type=str, default=r'outputs/infer/pcb-dataset/result', help="Path to the directory where output images will be saved.")                                # new output directory parameter
    args = parser.parse_args()
    main(args)


