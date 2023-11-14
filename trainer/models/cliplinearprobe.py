import torch
from clip import clip

from trainer import MODEL_REGISTERY, Trainer


@MODEL_REGISTERY.register()
class CLIPLinearProbe(Trainer):
    def build_model(self):
        print("Build CLIP LinearProbe")
        exit()
