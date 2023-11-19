import argparse
import torch

from trainer import build_trainer
from utils import collect_env_info, get_cfg_default, set_random_seed, setup_logger  # noqa


def reset_cfg_from_args(cfg, args):
    # ====================
    # Reset Global CfgNode
    # ====================
    cfg.GPU = args.gpu
    cfg.OUTPUT_DIR = args.output_dir
    cfg.SEED = args.seed
    cfg.DATASET.ROOT = args.root

    # ====================
    # Reset Dataset CfgNode
    # ====================
    if args.dataset:
        cfg.DATASET.NAME = args.dataset
    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains
    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    # ====================
    # Reset Model CfgNode
    # ====================
    if args.model:
        cfg.MODEL.NAME = args.model


def clean_cfg(cfg, model):
    """Remove Unused Model Configs


    Args:
        cfg (_C): Config Node.
        model (str): model name.
    """
    keys = list(cfg.MODEL.keys())
    for key in keys:
        if key == "NAME" or key == model:
            continue
        cfg.MODEL.pop(key, None)


def setup_cfg(args):
    cfg = get_cfg_default()

    if args.model_config_file:
        cfg.merge_from_file(args.model_config_file)

    reset_cfg_from_args(cfg, args)

    clean_cfg(cfg, args.model)

    print(cfg)
    exit()

    cfg.freeze()

    return cfg


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def main(args):
    cfg = setup_cfg(args)

    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    torch.cuda.set_device(cfg.GPU)

    setup_logger(cfg.OUTPUT_DIR)

    # print("*** Config ***")
    # print_args(args, cfg)

    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)
    if args.model == "CLIPZeroshot":
        trainer.test()
    else:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=str, default="./data/")
    parser.add_argument("--output-dir", type=str, default="./output/")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--source-domains", type=str, nargs="+")
    parser.add_argument("--target-domains", type=str, nargs="+")
    parser.add_argument("--model", type=str)
    parser.add_argument("--model-config-file", type=str)
    args = parser.parse_args()
    main(args)
