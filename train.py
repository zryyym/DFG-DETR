"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys

import swanlab

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

debug=False

if debug:
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

def main(args, ) -> None:
    """main
    """
    if args.test_only:
        dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed, deterministic=False)
    else:
        dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed, deterministic=True)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'


    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)

    if cfg.yaml_cfg['test_only']:
        val_ann_file = cfg.yaml_cfg['val_dataloader']['dataset']['ann_file']
        root, file = os.path.split(val_ann_file)
        file = 'instances_test.json'
        cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'] = os.path.join(root, file)

    if args.resume or args.tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == '__main__':
    names = []
    for name in names:
        parser = argparse.ArgumentParser()
        root = ''
        config = ''
        resume = None
        test_only = False

        base_name = os.path.basename(config).split('.')[0]
        base_dir = ''

        if resume is not None and not test_only:
            # ✅ 续训：直接使用 resume 所在文件夹作为 save_dir
            save_dir = os.path.dirname(resume)
        elif test_only:
            # 测试阶段直接使用基础路径
            save_dir = os.path.join(base_dir, base_name)
        else:
            # 训练阶段检查路径并添加数字后缀
            save_dir = os.path.join(base_dir, base_name)
            counter = 1
            while os.path.exists(save_dir):
                save_dir = os.path.join(base_dir, f"{base_name}_{counter}")
                counter += 1

        # priority 0
        parser.add_argument('-c', '--config', default=config, type=str, required=False)
        parser.add_argument('-r', '--resume', default=resume, type=str, help='resume from checkpoint')
        parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
        parser.add_argument('-d', '--device', type=str, help='device',)
        parser.add_argument('--seed', default=0, type=int, help='exp reproducibility')
        parser.add_argument('--use-amp', default=False, action='store_true', help='auto mixed precision training')
        parser.add_argument('--output-dir', default=save_dir, type=str, help='output directoy')
        parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
        parser.add_argument('--test-only', action='store_true', default=test_only,)

        # priority 1
        parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

        # env
        parser.add_argument('--print-method', type=str, default='builtin', help='print method')
        parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

        parser.add_argument('--local-rank', default=0, type=int, help='local rank id')
        args = parser.parse_args()

        try:
            main(args)
        finally:
            if swanlab.run is not None and not test_only:
                swanlab.finish()
