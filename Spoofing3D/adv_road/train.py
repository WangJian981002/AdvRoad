
import argparse
import sys

from Spoofing3D.adv_road.utils.train_utils import Training_pure_GAN, Training


def run_train(args):
    train_map = {
        'pure': Training_pure_GAN,
        'train': Training
    }

    train_type = args.type
    if train_type not in train_map:
        raise ValueError(f"Unknown train type: {train_type}. Available: {list(train_map.keys())}")

    train_func = train_map[train_type]

    # 构造 sys.argv
    sys.argv = [
        'train_inner.py',
        args.config,
        args.checkpoint,
    ]


    # 调用
    train_func()


def parse_train_args():
    """外层命令行解析"""
    parser = argparse.ArgumentParser(description="Unified trainuation runner for Spoofing3D models")

    parser.add_argument('--type', type=str, default='train', choices=['pure', 'train'])
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--score_thr', type=float, default=0.1)
    parser.add_argument('--iou_thr', nargs='+', type=float, default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument('--center_thr', nargs='+', type=float, default=[0.5, 1, 1.5, 2, 3])
    parser.add_argument('--is_4D', default='False',action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_train_args()
    run_train(args)
