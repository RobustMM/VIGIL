import os

import torch.nn as nn
from clip import clip

from trainer import MODEL_REGISTRY, Trainer


# TODO: CustomCLIP
class CustomCLIP(nn.Module):
    def __init__(self, cfg, class_names, clip_model):
        super().__init__()
        print("Hi")
        exit()


@MODEL_REGISTRY.register()
class CoCoOp(Trainer):
    # TODO: CoCoOp - Build_Model
    def build_model(self):
        print("Loading CLIP Backbone: {}".format(self.cfg.MODEL.CoCoOp.BACKBONE))
        clip_model, _ = clip.load(
            self.cfg.MODEL.CoCoOp.BACKBONE,
            device="cpu",
            download_root=os.path.abspath(os.path.expanduser("data")),
        )

        print("Building Custom CLIP")
        self.model = CustomCLIP(
            self.cfg, self.data_manager.dataset.class_names, clip_model
        )
