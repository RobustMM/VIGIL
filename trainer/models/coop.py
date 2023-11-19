import torch.nn as nn
from clip import clip

from trainer import MODEL_REGISTERY, Trainer


class PromptLearner(nn.Module):
    def __init__(self, cfg, class_names, clip_model):
        super().__init__()
        n_cls = len(class_names)
        n_ctx = cfg.MODEL.CoOp.N_CTX
        ctx_init = cfg.MODEL.CoOp.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imgsize = clip_model.visual.input_resolution
        cfg_imgsize = cfg.INPUT.SIZE[0]

        # Random Initialization for Context Vectors
        if cfg.MODEL.CoOp.CSC:
            print("Initializing Class-Specific Contexts")
        else:
            print("Initializing a Unified Context")

        exit()


class CustomCLIP(nn.Module):
    def __init__(self, cfg, class_names, clip_model):
        super().__init__()

        # TODO: Build Prompt_Learner
        self.prompt_learner = PromptLearner(cfg, class_names, clip_model)

        exit()


@MODEL_REGISTERY.register()
class CoOp(Trainer):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

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
