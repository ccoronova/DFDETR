"""
Dataset registry center -- Linux version
When switching machines, only edit the paths in this file
"""

DATASETS = {
    'pcb': {
        'num_classes': 6,
        'train_img_folder': r'/home/user/user/rtdetr code/RT-DETR-main/PCB_DATASET/train',
        'train_ann_file': r'/home/user/user/rtdetr code/RT-DETR-main/PCB_DATASET/train.json',
        'val_img_folder': r'/home/user/user/rtdetr code/RT-DETR-main/PCB_DATASET/val',
        'val_ann_file': r'/home/user/user/rtdetr code/RT-DETR-main/PCB_DATASET/val.json',
        'categories': {
            1: 'missing_hole',
            2: 'mouse_bite',
            3: 'open_circuit',
            4: 'short',
            5: 'spur',
            6: 'spurious_copper',
        },
        # ---- Inference visualization ----
        'weight_path': r'D:\user\detr code\rtdetr code-1\rtdetr code-1\out\pacdataset\best_map50.pth',
        'im_dir': r'd:\user\detr code\three\infer\pcb-dataset',
        'output_dir': r'D:\user\detr code\rtdetr code-1\rtdetr code-1\out\pacdataset\you\infer-chu',
        'label_mapping': {
            0: 'missing_hole',
            1: 'mouse_bite',
            2: 'open_circuit',
            3: 'short',
            4: 'spur',
            5: 'spurious_copper',
        },
        'color_mapping': {
            0: "#FF0000",
            1: "#00FF00",
            2: "#0000FF",
            3: "#FFFF00",
            4: "#FF00FF",
            5: "#00FFFF",
        },
        'font_size': 80,
        'draw_thresh': 0.46,
    },

    'deeppcb': {
        'num_classes': 6,
        'train_img_folder': r'/home/user/user/rtdetr code/Deeppcb/dataset_coco/train/images/',
        'train_ann_file': r'/home/user/user/rtdetr code/Deeppcb/dataset_coco/annotations/train_coco_format.json',
        'val_img_folder': r'/home/user/user/rtdetr code/Deeppcb/dataset_coco/val/images/',
        'val_ann_file': r'/home/user/user/rtdetr code/Deeppcb/dataset_coco/annotations/val_coco_format.json',
        'categories': {
            1: 'open',
            2: 'short',
            3: 'mousebite',
            4: 'spur',
            5: 'copper',
            6: 'pin-hole',
        },
        # ---- Inference visualization ----
        'weight_path': r'D:\user\detr code\rtdetr code-1\rtdetr code-1\out\deeppcb\best_map50.pth',
        'im_dir': r'd:\user\detr code\three\infer\deeppcb',
        'output_dir': r'D:\user\detr code\rtdetr code-1\rtdetr code-1\out\deeppcb\you\infer1',
        'label_mapping': {
            0: 'open',
            1: 'short',
            2: 'mousebite',
            3: 'spur',
            4: 'copper',
            5: 'pin-hole',
        },
        'color_mapping': {
            0: "#0000FF",  # open - blue
            1: "#FF00FF",  # short - magenta
            2: "#00FF00",  # mousebite - green
            3: "#00FFFF",  # spur - cyan
            4: "#FF0000",  # copper - red
            5: "#FFFF00",  # pin-hole - yellow
        },
        'font_size': 30,
        'draw_thresh': 0.8,
    },

    'pcb-aol': {
        'num_classes': 2,
        'train_img_folder': r'/home/user/user/rtdetr code1/RT-DETR-main/pcb-aol-dataset/train/images-jpg',
        'train_ann_file': r'/home/user/user/rtdetr code1/RT-DETR-main/pcb-aol-dataset/train.json',
        'val_img_folder': r'/home/user/user/rtdetr code1/RT-DETR-main/pcb-aol-dataset/val/images-jpg/',
        'val_ann_file': r'/home/user/user/rtdetr code1/RT-DETR-main/pcb-aol-dataset/val.json',
        'categories': {
            1: 'Bad_podu',
            2: 'Bad_qiaojiao',
        },
        'weight_path': '',  # TODO
        'im_dir': '',   # TODO
        'output_dir': '',  # TODO
        'label_mapping': {
            0: 'Bad_podu',
            1: 'Bad_qiaojiao',
        },
        'color_mapping': {
            0: "#FF0000",
            1: "#00FF00",
        },
        'font_size': 80,
        'draw_thresh': 0.46,
    },

    'neu-det': {
        'num_classes': 6,
        'train_img_folder': r'/home/user/user/rtdetr code1/RT-DETR-main/NEU-DET-COCO/NEU-DET-COCO/train/',
        'train_ann_file': r'/home/user/user/rtdetr code1/RT-DETR-main/NEU-DET-COCO/NEU-DET-COCO/annotations/train.json',
        'val_img_folder': r'/home/user/user/rtdetr code1/RT-DETR-main/NEU-DET-COCO/NEU-DET-COCO/val/',
        'val_ann_file': r'/home/user/user/rtdetr code1/RT-DETR-main/NEU-DET-COCO/NEU-DET-COCO/annotations/val.json',
        'categories': {
            1: 'Crazing',
            2: 'Inclusion',
            3: 'Patches',
            4: 'Pitted Surface',
            5: 'Rolled_in Scale',
            6: 'Scratches',
        },
        'weight_path': '',  # TODO
        'im_dir': '',   # TODO
        'output_dir': '',  # TODO
        'label_mapping': {
            0: 'Crazing',
            1: 'Inclusion',
            2: 'Patches',
            3: 'Pitted Surface',
            4: 'Rolled_in Scale',
            5: 'Scratches',
        },
        'color_mapping': {
            0: "#FF0000",
            1: "#00FF00",
            2: "#0000FF",
            3: "#FFFF00",
            4: "#FF00FF",
            5: "#00FFFF",
        },
        'font_size': 80,
        'draw_thresh': 0.46,
    },

    'new-gc-net': {
        'num_classes': 10,
        'train_img_folder': r'/home/user/user/rtdetr code1/RT-DETR-main/New_GC-DET/train/',
        'train_ann_file': r'/home/user/user/rtdetr code1/RT-DETR-main/New_GC-DET/train.json',
        'val_img_folder': r'/home/user/user/rtdetr code1/RT-DETR-main/New_GC-DET/val/',
        'val_ann_file': r'/home/user/user/rtdetr code1/RT-DETR-main/New_GC-DET/val.json',
        'categories': {
            1: 'chongkong',
            2: 'hanfeng',
            3: 'yueyawan',
            4: 'shuiban',
            5: 'youban',
            6: 'siban',
            7: 'yiwu',
            8: 'yahen',
            9: 'zenhen',
            10: 'yaozhe',
        },
        'weight_path': '',  # TODO
        'im_dir': '',   # TODO
        'output_dir': '',  # TODO
        'label_mapping': {
            0: 'chongkong', 1: 'hanfeng', 2: 'yueyawan', 3: 'shuiban', 4: 'youban',
            5: 'siban', 6: 'yiwu', 7: 'yahen', 8: 'zenhen', 9: 'yaozhe',
        },
        'color_mapping': {
            0: "#FF0000", 1: "#00FF00", 2: "#0000FF", 3: "#FFFF00", 4: "#FF00FF",
            5: "#00FFFF", 6: "#FFA500", 7: "#800080", 8: "#008080", 9: "#FFC0CB",
        },
        'font_size': 80,
        'draw_thresh': 0.46,
    },

    'apspc': {
        'num_classes': 0,  # TODO: fill in num_classes + categories
        'train_img_folder': r'/home/user/user/rtdetr code1/RT-DETR-main/APSPC-COCO/coco/images/train/',
        'train_ann_file': r'/home/user/user/rtdetr code1/RT-DETR-main/APSPC-COCO/coco/annotations/instances_train.json',
        'val_img_folder': r'/home/user/user/rtdetr code1/RT-DETR-main/APSPC-COCO/coco/images/val/',
        'val_ann_file': r'/home/user/user/rtdetr code1/RT-DETR-main/APSPC-COCO/coco/annotations/instances_val.json',
        'categories': {},  # TODO
        'weight_path': '',  # TODO
        'im_dir': '',   # TODO
        'output_dir': '',  # TODO
        'label_mapping': {},  # TODO
        'color_mapping': {},  # TODO
        'font_size': 80,
        'draw_thresh': 0.46,
    },

    'pcb-defect': {
        'num_classes': 0,  # TODO: fill in num_classes + categories
        'train_img_folder': r'/home/user/user/rtdetr code1/RT-DETR-main/pcb-defect-dataset/train/images',
        'train_ann_file': r'/home/user/user/rtdetr code1/RT-DETR-main/pcb-defect-dataset/annotations/instances_train.json',
        'val_img_folder': r'/home/user/user/rtdetr code1/RT-DETR-main/pcb-defect-dataset',  # TODO: confirm the val image directory
        'val_ann_file': r'/home/user/user/rtdetr code1/RT-DETR-main/pcb-defect-dataset/annotations/instances_val.json',
        'categories': {},  # TODO
        'weight_path': '',  # TODO
        'im_dir': '',   # TODO
        'output_dir': '',  # TODO
        'label_mapping': {},  # TODO
        'color_mapping': {},  # TODO
        'font_size': 80,
        'draw_thresh': 0.46,
    },
}


def get_dataset(name: str):
    """Return the dataset config; raise an error with available names if not found"""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}', available: {list(DATASETS.keys())}")
    return DATASETS[name]
