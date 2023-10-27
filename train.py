import argparse
import torch

from utils.config_default import get_cfg_default


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=str,
        default="0"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--dataset",
        type=str
    )
    parser.add_argument(
        "--source_domain",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--target_domain",
        type=str
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./data/"
    )
    parser.add_argument(
        "--outputs",
        type=str,
        default="./output/"
    )
    parser.add_argument(
        "--model",
        type=str
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=10
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5
    )

    return parser.parse_args()


def setup_cfg(args):
    cfg = get_cfg_default()

    return cfg


def main(args):
    # torch.set_num_threads(1)  # This limits threads to avoid server crash

    cfg = setup_cfg(args)
    print(cfg)


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
