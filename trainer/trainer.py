import torch

from collections import OrderedDict


class BaseTrainer():
    """Base Class for Iterative Trainer."""

    def __init__(self):
        self._models = OrderedDict()
        self._optimizers = OrderedDict()
        self._schedulers = OrderedDict()
        self._writer = None


class GenericTrainer():
    """Generic Trainer Class for Implementing Generic Function"""

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.cuda.current_device()
        self.start_epoch = 0
        self.max_epoch = cfg.OPTIM.EPOCH
        self.output_dir = cfg.OUTPUTS
        self.cfg = cfg

        # Build DataLoader
        print("Build DataLoader")

        # Build Model
        self.build_model()
