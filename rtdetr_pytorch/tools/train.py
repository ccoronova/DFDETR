"""by lyuwenyu
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS
import warnings
import numpy as np
import random

# ============================================================
# Switch dataset: just change this one variable (options in dataset_registry.py)
# ============================================================
DATASET = 'pcb'  # pcb | deeppcb | pcb-aol | neu-det | new-gc-net


# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*image size.*")


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    # ---------- Inject the dataset path & categories from the DATASET variable ----------
    from configs.dataset.dataset_registry import get_dataset
    from src.data.coco.coco_dataset import set_dataset_categories

    ds = get_dataset(DATASET)
    set_dataset_categories(ds['categories'])

    # Inject the paths and num_classes into yaml_cfg (deep merge)
    cfg = YAMLConfig(
        args.config,
        resume=args.resume,
        use_amp=args.amp,
        tuning=args.tuning
    )
    cfg.yaml_cfg['num_classes'] = ds['num_classes']
    cfg.yaml_cfg['train_dataloader']['dataset']['img_folder'] = ds['train_img_folder']
    cfg.yaml_cfg['train_dataloader']['dataset']['ann_file'] = ds['train_ann_file']
    cfg.yaml_cfg['val_dataloader']['dataset']['img_folder'] = ds['val_img_folder']
    cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'] = ds['val_ann_file']

    # Configure the model and optimizer according to the task
    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        # If it is a validation task, run validation
        solver.val()
    else:
        # Run the training task
        solver.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default=r'configs/rtdetr/rtdetr_r18vd_6x_coco.yml')
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, help='seed')  # Set the default random seed to 0

    args = parser.parse_args()
  
    main(args)
