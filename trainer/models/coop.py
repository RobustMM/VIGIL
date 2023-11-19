import torch.nn as nn
from clip import clip

from trainer import MODEL_REGISTERY, Trainer


class CustomCLIP(nn.Module):
    def __init__(self, cfg, class_names, clip_model):
        super().__init__()
        print("Hi")
        exit()


@MODEL_REGISTERY.register()
class CoOp(Trainer):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        pass

    def build_model(self):
        clip_model, _ = clip.load(
            self.cfg.MODEL.CoOp.BACKBONE,
            device="cpu",
            download_root="/data/dzha866/Project/VIGIL/data/",
        )

        print("Building Custom CLIP")
        self.model = CustomCLIP(
            self.cfg, self.data_manager.dataset.class_names, clip_model
        )

        exit()
