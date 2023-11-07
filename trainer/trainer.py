import time

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from datasets import DataManager
from evaluator import build_evaluator


class Trainer:
    """Generic Trainer Class for Implementing Generic Function"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR
        self.device = torch.cuda.current_device()

        self._writer = None

        # Build Data Manager
        self.data_manager = DataManager(self.cfg)
        self.data_loader_train = self.data_manager.data_loader_train
        self.data_loader_val = self.data_manager.data_loader_val
        self.data_loader_test = self.data_manager.data_loader_test
        self.num_classes = self.data_manager.num_classes
        self.class_label_name_mapping = self.data_manager.class_label_name_mapping

        # Build Model
        self.build_model()

        # Build Evaluator
        self.evaluator = build_evaluator(
            cfg, class_label_name_mapping=self.class_label_name_mapping
        )

    def build_model(self):
        raise NotImplementedError

    def detect_abnormal_loss(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is Infinite or NaN.")

    def init_writer(self, log_dir):
        if self._writer is None:
            print("Initializing Summary Writer with log_dir={}".format(log_dir))
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is not None:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self):
        self.before_train()
        for self.current_epoch in range(self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def before_train(self):
        # Initialize SummaryWriter
        # writer_dir = osp.join(self.output_dir, "tensorboard")
        # mkdir_if_missing(writer_dir)
        # self.init_writer(writer_dir)
        self.time_start = time.time()

    def after_train(self):
        # self.save_model()
        self.test()

    # TODO: run_epoch()
    def run_epoch(self):
        print(self.current_epoch)

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    @torch.no_grad()
    def test(self, split=None):
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "Validation" and self.val_loader is not None:
            data_loader = self.data_loader_val
        elif split == "Test":
            data_loader = self.data_loader_test
        else:
            raise NotImplementedError

        print("Evaluate on the {} Set".format(split))

        for _, batch_data in enumerate(tqdm(data_loader)):
            input_data, class_label = self.parse_batch_test(batch_data)
            output = self.model_inference(input_data)
            print(output)
            print(len(output))
            exit()

    def parse_barch_train(self, batch_data):
        raise NotImplementedError

    def parse_batch_test(self, batch_data):
        input_data = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)
        return input_data, class_label

    def forward_backward(self, batch_data):
        raise NotImplementedError

    def model_inference(self, input_data):
        raise NotImplementedError

    def get_current_lr(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError
