import argparse

from trainer import build_trainer
from utils import get_cfg_default, set_random_seed, set_device, setup_logger


def reset_cfg_from_args(cfg, args):
    # ====================
    # Reset Global CfgNode
    # ====================
    cfg.GPU = args.gpu
    cfg.OUTPUTS = args.outputs
    cfg.SEED = args.seed
    cfg.DATASET.ROOT = args.root

    # ====================
    # Reset Dataset CfgNode
    # ====================
    if args.dataset:
        cfg.DATASET.NAME = args.dataset
    if args.source_domain:
        cfg.DATASET.SOURCE_DOMAIN = args.source_domain
    if args.target_domain:
        cfg.DATASET.TARGET_DOMAIN = args.target_domain

    # ====================
    # Reset DataLoader CfgNode
    # ====================
    cfg.DATALOADER.TRAIN.BATCH_SIZE = args.batch_size

    # ====================
    # Reset Model CfgNode
    # ====================
    if args.model:
        cfg.MODEL.NAME = args.model

    # ====================
    # Reset Optimizer CfgNode
    # ====================
    cfg.OPTIM.EPOCH = args.epoch
    cfg.OPTIM.LR = args.lr


def setup_cfg(args):
    cfg = get_cfg_default()
    print(cfg)
    exit()

    reset_cfg_from_args(cfg, args)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)

    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    set_device(cfg.GPU)
    setup_logger(cfg.OUTPUTS)

    # print("*** Config ***")
    # print(cfg)

    trainer = build_trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=int,
        default=0
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./data/"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/"
    )
    parser.add_argument(
        "--dataset",
        type=str
    )
    parser.add_argument(
        "--source-domains",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--target-domains",
        type=str,
        nargs="+"
    )
    parser.add_argument(
        "--model",
        type=str
    )
    parser.add_argument(
        "--backbone",
        type=str
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=10
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5
    )
    args = parser.parse_args()
    main(args)
