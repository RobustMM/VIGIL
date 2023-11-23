import torch
from torch.optim.lr_scheduler import _LRScheduler

AVAILABLE_LR_SCHEDULERS = ["cosine"]


class _BaseWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, scheduler, warmup_epoch, last_epoch=-1):
        self.scheduler = scheduler
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.scheduler.step(epoch)
            self._last_lr = self.scheduler.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):
    def __init__(self, optimizer, scheduler, warmup_epoch, cons_lr, last_epoch=-1):
        self.cons_lr = cons_lr
        super().__init__(optimizer, scheduler, warmup_epoch, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.scheduler.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


def build_lr_scheduler(optimizer, optim_cfg):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        optim_cfg (CfgNode): optimization config.
    """

    if optim_cfg.LR_SCHEDULER not in AVAILABLE_LR_SCHEDULERS:
        raise ValueError(
            "LR Scheduler must be one of {}, but got {}".format(
                AVAILABLE_LR_SCHEDULERS, optim_cfg.LR_SCHEDULER
            )
        )

    if optim_cfg.LR_SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=optim_cfg.MAX_EPOCH
        )

    if optim_cfg.WARMUP_TYPE == "constant":
        scheduler = ConstantWarmupScheduler(
            optimizer, scheduler, optim_cfg.WARMUP_EPOCH, optim_cfg.WARMUP_CONS_LR
        )

    return scheduler
