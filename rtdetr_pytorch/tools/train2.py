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
# Read the configuration
    cfg = YAMLConfig(
        args.config,
        resume=args.resume,
        use_amp=args.amp,
        tuning=args.tuning
    )
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
    parser.add_argument('--seed', type=int, help='seed',)
    args = parser.parse_args()

    main(args)
