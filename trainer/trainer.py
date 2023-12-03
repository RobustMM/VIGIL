import datetime
import os
import time
from collections import OrderedDict

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import DataManager
from evaluator import build_evaluator
from utils import AverageMeter, MetricMeter, save_checkpoint


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

        self._models = OrderedDict()
        self._optimizers = OrderedDict()
        self._lr_schedulers = OrderedDict()

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

    def model_zero_grad(self, model_names=None):
        model_names = self.get_model_names(model_names)
        for model_name in model_names:
            if self._optimizers[model_name] is not None:
                self._optimizers[model_name].zero_grad()

    def model_backward(self, loss):
        self.detect_abnormal_loss(loss)
        loss.backward()

    def model_update(self, model_names=None):
        model_names = self.get_model_names(model_names)
        for model_name in model_names:
            if self._optimizers[model_name] is not None:
                self._optimizers[model_name].step()

    def model_backward_and_update(self, loss, model_names=None):
        self.model_zero_grad(model_names)
        self.model_backward(loss)
        self.model_update(model_names)

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

    def run_epoch(self):
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.data_loader_train)
        end_time = time.time()

        for self.batch_idx, batch_data in enumerate(self.data_loader_train):
            data_time.update(time.time() - end_time)
            loss_summary = self.forward_backward(batch_data)
            batch_time.update(time.time() - end_time)
            losses.update(loss_summary)

            if (
                (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
                or self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            ):
                num_batches_remain = 0
                num_batches_remain += self.num_batches - self.batch_idx - 1
                num_batches_remain += (
                    self.max_epoch - self.current_epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * num_batches_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.current_epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"{losses}"]
                info += [f"lr {self.optimizer.param_groups[0]['lr']:.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            end_time = time.time()

    def before_epoch(self):
        pass

    def after_epoch(self):
        if self.current_epoch + 1 == self.max_epoch:
            self.save_model(self.current_epoch, self.output_dir)

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
            self.evaluator.process(output, class_label)

        evaluation_results = self.evaluator.evaluate()

        return list(evaluation_results.values())[0]

    def parse_batch_train(self, batch_data):
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

    def model_registeration(
        self, model_name="model", model=None, optimizer=None, lr_scheduler=None
    ):
        assert model_name not in self._models, "Found duplicate model names."

        self._models[model_name] = model
        self._optimizers[model_name] = optimizer
        self._lr_schedulers[model_name] = lr_scheduler

    def get_model_names(self, model_names=None):
        if model_names is not None:
            for model_name in model_names:
                assert model_name in list(self._models.keys())
        else:
            return list(self._models.keys())

    # TODO: Save_Model
    def save_model(
        self,
        current_epoch,
        save_dir,
        model_name="name",
    ):
        model_names = self.get_model_names()

        for model_name in model_names:
            model_dict = self._models[model_name].state_dict()

            optimizer_dict = None
            if self._optimizers[model_name] is not None:
                optimizer_dict = self._optimizers[model_name].state_dict()

            lr_scheduler_state_dict = None
            if self._lr_schedulers[model_name] is not None:
                lr_scheduler_state_dict = self._lr_schedulers[model_name].state_dict()

            # Remove "module." in state_dict's keys
            new_model_dict = OrderedDict()
            for key, value in model_dict.items():
                if key.startswith("module."):
                    key = key[7:]
                new_model_dict[key] = value
            model_dict = new_model_dict

            # Save Model
            if not model_name:
                model_name = "model.pth.tar-" + str(current_epoch)
            
            fpath = os.path.join(save_dir, model_name)
