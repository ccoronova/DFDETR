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
# 切换数据集：改这一个变量即可（可选值见 dataset_registry.py）
# ============================================================
DATASET = 'pcb'  # pcb | deeppcb | pcb-aol | neu-det | new-gc-net


# 忽略特定的警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*image size.*")


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    # ---------- 根据 DATASET 变量注入数据集路径 & 类别 ----------
    from configs.dataset.dataset_registry import get_dataset
    from src.data.coco.coco_dataset import set_dataset_categories

    ds = get_dataset(DATASET)
    set_dataset_categories(ds['categories'])

    # 把路径和 num_classes 注入到 yaml_cfg 中（deep merge）
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
#根据任务配置模型，优化器
    
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
   
    if args.test_only:
        #如果是验证任务，执行val
        solver.val()  
    else:
        #执行训练任务
        solver.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default=r'configs/rtdetr/rtdetr_r18vd_6x_coco.yml')
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, help='seed')  # 设置默认随机种子为0

    args = parser.parse_args()
  
    main(args)
