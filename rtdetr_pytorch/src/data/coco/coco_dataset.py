"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

import torch
import torch.utils.data

import torchvision
torchvision.disable_beta_transforms_warning()

from torchvision import datapoints

from pycocotools import mask as coco_mask

from ...core import register

__all__ = ['CocoDetection']


@register
class CocoDetection(torchvision.datasets.CocoDetection):
    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category']
    
    def __init__(self, img_folder, ann_file, transforms, return_masks, remap_mscoco_category=False):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, remap_mscoco_category)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

         # # Build mscoco_category2label alphabetically
       #  sorted_categories = sorted(mscoco_category2name.items(), key=lambda x: x[1])
       #  self.mscoco_category2label = {cat_id: idx for idx, (cat_id, _) in enumerate(sorted_categories)}
       #  # Print the category order
       #  print("Training category order: ")
       #  for cat_id, cat_name in sorted(mscoco_category2name.items(), key=lambda x: x[1]):
       #      print(f"ID: {cat_id}, Name: {cat_name}, Label: {mscoco_category2label[cat_id]}")
    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # ['boxes', 'masks', 'labels']:
        if 'boxes' in target:
            target['boxes'] = datapoints.BoundingBox(
                target['boxes'], 
                format=datapoints.BoundingBoxFormat.XYXY, 
                spatial_size=img.size[::-1]) # h w

        if 'masks' in target:
            target['masks'] = datapoints.Mask(target['masks'])

        if self._transforms is not None:
            img, target = self._transforms(img, target)
            
        return img, target

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'

        return s 


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, remap_mscoco_category=False):
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.remap_mscoco_category:
            classes = [mscoco_category2label[obj["category_id"]] for obj in anno]
        else:
            classes = [obj["category_id"] for obj in anno]
        #
        # if self.remap_mscoco_category:
        # use the mscoco_category2label defined in the CocoDetection class
        #     classes = [self.mscoco_category2label[obj["category_id"]] for obj in anno]
        # else:
        #     classes = [obj["category_id"]] for obj in anno]

        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        target["size"] = torch.as_tensor([int(w), int(h)])
    
        return image, target


mscoco_category2name = {}  # injected by set_dataset_categories() or train.py
mscoco_category2label = {}
mscoco_label2category = {}


def set_dataset_categories(categories: dict):
    """Called by train.py before training to inject the current dataset category mapping"""
    global mscoco_category2name, mscoco_category2label, mscoco_label2category
    mscoco_category2name = dict(categories)
    mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
    mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}

    # Keep the __init__.py references in sync, otherwise rtdetr_postprocessor still sees an empty {}
    import src.data.coco as coco_pkg
    coco_pkg.mscoco_category2name = mscoco_category2name
    coco_pkg.mscoco_category2label = mscoco_category2label
    coco_pkg.mscoco_label2category = mscoco_label2category
