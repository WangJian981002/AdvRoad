


import argparse
import sys
from Spoofing3D.adv_road.utils.eval_utils import eval, eval_two_stage, eval_LPIPS


def run_eval(args):
    eval_map = {
        'eval': eval,
        'two_stage': eval_two_stage,
        'lpips': eval_LPIPS
    }

    eval_type = args.type
    if eval_type not in eval_map:
        raise ValueError(f"Unknown eval type: {eval_type}. Available: {list(eval_map.keys())}")

    eval_func = eval_map[eval_type]

    # 构造 sys.argv
    sys.argv = [
        'eval_inner.py',
        args.config,
        args.checkpoint,
    ]


    # 调用
    eval_func(
        G_dir=args.G_dir,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        center_thr=args.center_thr,
        is_4D=args.is_4D,
    )


def parse_eval_args():
    """外层命令行解析"""
    parser = argparse.ArgumentParser(description="Unified evaluation runner for Spoofing3D models")

    parser.add_argument('--type', type=str, default='eval', choices=['eval', 'two_stage', 'lpips'])
    parser.add_argument('--G_dir', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--score_thr', type=float, default=0.1)
    parser.add_argument('--iou_thr', nargs='+', type=float, default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument('--center_thr', nargs='+', type=float, default=[0.5, 1, 1.5, 2, 3])
    parser.add_argument('--is_4D', default='False',action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_eval_args()
    run_eval(args)
