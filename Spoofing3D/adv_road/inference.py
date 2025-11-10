import argparse
import sys

from Spoofing3D.adv_road.utils.inference_utils import *


def run_inference(args):
    inference_map = {
        'inference': inference,
        'whole': inference_whole,
        'whole_poster': inference_whole_poster
    }

    inference_type = args.type
    if inference_type not in inference_map:
        raise ValueError(f"Unknown inference type: {inference_type}. Available: {list(inference_map.keys())}")

    inference_func = inference_map[inference_type]

    # 构造 sys.argv
    sys.argv = [
        'inference_inner.py',
        args.config,
        args.checkpoint,
    ]


    # 调用
    inference_func(
        ind = args.ind,
        path = args.path,
        is_4D=args.is_4D
    )


def parse_inference_args():
    """外层命令行解析"""
    parser = argparse.ArgumentParser(description="Unified inferenceuation runner for Spoofing3D models")

    parser.add_argument('--type', type=str, default='inference', choices=['inference', 'whole', 'whole_poster'])
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--ind', default='None')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--score_thr', type=float, default=0.1)
    parser.add_argument('--iou_thr', nargs='+', type=float, default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument('--center_thr', nargs='+', type=float, default=[0.5, 1, 1.5, 2, 3])
    parser.add_argument('--is_4D', default='False',action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_inference_args()
    run_inference(args)
