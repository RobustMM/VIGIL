import argparse

from utils import get_cfg_default, set_random_seed, set_device, setup_logger


def get_args():
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
        default="./outputs/"
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

    reset_cfg_from_args(cfg, args)

    cfg.freeze()

    return cfg


def main(args):
    # torch.set_num_threads(1)  # This limits threads to avoid server crash

    cfg = setup_cfg(args)

    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    set_device(cfg.GPU)
    setup_logger(cfg.OUTPUTS)

    # print("*** Config ***")
    # print(cfg)


if __name__ == "__main__":
    args = get_args()
    main(args)
