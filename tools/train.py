# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def add_early_stopping_hook(cfg, args):
    if args.early_stop_monitor is None:
        return

    early_stop_hook = dict(
        type='EarlyStoppingHook',
        monitor=args.early_stop_monitor,
        rule=args.early_stop_rule,
        min_delta=args.early_stop_min_delta,
        strict=args.early_stop_strict,
        check_finite=not args.early_stop_no_check_finite,
        patience=args.early_stop_patience)
    if args.early_stop_stopping_threshold is not None:
        early_stop_hook['stopping_threshold'] = \
            args.early_stop_stopping_threshold

    custom_hooks = []
    for hook in cfg.get('custom_hooks', []):
        if isinstance(hook, dict) and hook.get('type') == 'EarlyStoppingHook':
            continue
        custom_hooks.append(hook)

    custom_hooks.append(early_stop_hook)
    cfg.custom_hooks = custom_hooks


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--early-stop-monitor',
        type=str,
        help='validation metric used by EarlyStoppingHook, '
        'for example coco/segm_mAP')
    parser.add_argument(
        '--early-stop-patience',
        type=int,
        default=5,
        help='number of validation intervals with no improvement before '
        'stopping training')
    parser.add_argument(
        '--early-stop-min-delta',
        type=float,
        default=0.0,
        help='minimum metric improvement required to reset patience')
    parser.add_argument(
        '--early-stop-rule',
        choices=['greater', 'less'],
        default='greater',
        help='whether the monitored metric should increase or decrease')
    parser.add_argument(
        '--early-stop-stopping-threshold',
        type=float,
        help='optional target threshold that stops training once reached')
    parser.add_argument(
        '--early-stop-strict',
        action='store_true',
        help='raise an error if the monitored metric is missing')
    parser.add_argument(
        '--early-stop-no-check-finite',
        action='store_true',
        help='disable NaN/Inf checks in EarlyStoppingHook')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    add_early_stopping_hook(cfg, args)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
