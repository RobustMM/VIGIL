import os

import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer

from trainer import MODEL_REGISTRY, Trainer

_tokenizer = SimpleTokenizer()


class PromptLearner(nn.Module):
    def __init__(self, cfg, class_names, clip_model):
        super().__init__()
        self.n_cls = len(class_names)
        self.n_ctx = cfg.MODEL.CoOp.N_CTX
        ctx_dim = clip_model.ln_final.weight.shape[0]

        assert (
            cfg.INPUT.SIZE[0] == clip_model.visual.input_resolution
        ), "Input Size {} must Equal to CLIP Image Encoder Input Resolution {}".format(
            cfg.INPUT.SIZE,
            (clip_model.visual.input_resolution, clip_model.visual.input_resolution),
        )

        # Random Initialization for Context Vectors
        if cfg.MODEL.CoOp.CSC:
            print("Initializing Class-Specific Contexts")
            ctx_vectors = torch.empty(
                self.n_cls, self.n_ctx, ctx_dim, dtype=clip_model.dtype
            )
        else:
            print("Initializing a Unified Context")
            ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=clip_model.dtype)

        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * self.n_ctx)
        print("Initial Context: {}".format(prompt_prefix))
        print("Number of Context Tokens: {}".format(self.n_ctx))
        self.ctx = nn.Parameter(ctx_vectors)  # To be optimized

        class_names = [class_name.replace("_", " ") for class_name in class_names]
        self.class_name_lens = [
            len(_tokenizer.encode(class_name)) for class_name in class_names
        ]
        prompts = [prompt_prefix + " " + class_name + "." for class_name in class_names]
        self.prompts_tokenized = torch.cat(
            [clip.tokenize(prompt) for prompt in prompts]
        )
        self.class_token_position = cfg.MODEL.CoOp.CLASS_TOKEN_POSITION

    def forward(self):
        print("Prompt Learner Forward")
        exit()


class CustomCLIP(nn.Module):
    def __init__(self, cfg, class_names, clip_model):
        super().__init__()

        self.prompt_learner = PromptLearner(cfg, class_names, clip_model)
        self.promptes_tokenized = self.prompt_learner.prompts_tokenized
        self.image_encoder = clip_model.visual
        # TODO: Build TextEncoder
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        exit()

    def forward(self, image):
        print("Custom CLIP Forward")
        exit()


@MODEL_REGISTRY.register()
class CoOp(Trainer):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def build_model(self):
        clip_model, _ = clip.load(
            self.cfg.MODEL.CoOp.BACKBONE,
            device="cpu",
            download_root=os.path.abspath(os.path.expanduser("data")),
        )
        clip_model.half()

        print("Building Custom CLIP")
        self.model = CustomCLIP(
            self.cfg, self.data_manager.dataset.class_names, clip_model
        )

        exit()
