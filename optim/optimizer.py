import torch
import torch.nn as nn

AVAI_OPTIMS = ["sgd"]


def build_optimizer(model, optim_cfg, param_groups=None):
    """A function wrapper for building an optimizer.

    Args:
        model (nn.Module or iterable): model.
        optim_cfg (CfgNode): optimization config.
        param_groups: If provided, directly optimize param_groups and abandon model
    """
    if optim_cfg.NAME not in AVAI_OPTIMS:
        raise ValueError(
            "optim must be one of {}, but got {}".format(AVAI_OPTIMS, optim_cfg.NAME)
        )

    if isinstance(model, nn.Module):
        param_groups = model.parameters()
    else:
        param_groups = model

    if optim_cfg.NAME == "sgd":
        optimizer = torch.optim.SGD(
            params=param_groups,
            lr=optim_cfg.LR,
            momentum=optim_cfg.MOMENTUM,
            weight_decay=optim_cfg.WEIGHT_DECAY,
            dampening=optim_cfg.SGD_DAMPENING,
            nesterov=optim_cfg.SGD_NESTEROV,
        )

    return optimizer
