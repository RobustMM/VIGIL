import torch

from torch.utils.tensorboard import SummaryWriter


class Trainer():
    """Generic Trainer Class for Implementing Generic Function"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.epoch = cfg.OPTIM.EPOCH
        self.output_dir = cfg.OUTPUTS
        self.device = torch.cuda.current_device()
        self._writer = None

        # TODO: Build Data Manager

        # TODO: Build Model
        self.build_model()

        # TODO: Build Evaluator

    def detect_abnormal_loss(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is Infinite or NaN.")

    def init_writer(self, log_dir):
        if self._writer is None:
            print("Initializing Summary Writer with log_dir={}".format(log_dir))
            self._writer = SummaryWriter(log_dir=log_dir)

    def save_model(self, directory):
        raise NotImplementedError
