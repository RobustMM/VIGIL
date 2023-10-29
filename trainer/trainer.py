import torch


class Trainer():
    """Generic Trainer Class for Implementing Generic Function"""

    def __init__(self, cfg):

        self.device = torch.cuda.current_device()
        self.start_epoch = 0
        self.max_epoch = cfg.OPTIM.EPOCH
        self.output_dir = cfg.OUTPUTS
        self.cfg = cfg

        # TODO: Build Data Manager

        # TODO: Build Model
        self.build_model()

        # TODO: Build Evaluator
