import os
from collections import OrderedDict

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
        self.n_ctx = cfg.MODEL.CoCoOp.N_CTX

        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim

        # Random Initialization Context
        ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=clip_model.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * self.n_ctx)
        print("Initial Context: {}".format(prompt_prefix))
        print("Number of Context Tokens: {}".format(self.n_ctx))
        self.ctx = nn.Parameter(ctx_vectors)  # To be optimized

        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
                ]
            )
        )

        if cfg.MODEL.CoCoOp.PREC == "fp16":
            self.meta_net.half()

        class_names = [class_name.replace("_", " ") for class_name in class_names]
        self.class_name_lens = [
            len(_tokenizer.encode(class_name)) for class_name in class_names
        ]
        prompts = [prompt_prefix + " " + class_name + "." for class_name in class_names]
        self.prompts_tokenized = torch.cat(
            [clip.tokenize(prompt) for prompt in prompts]
        )

        with torch.no_grad():
            prompts_embedding = clip_model.token_embedding(self.prompts_tokenized).type(
                clip_model.dtype
            )

        self.register_buffer("token_prefix", prompts_embedding[:, :1, :])  # SOS
        self.register_buffer(
            "token_suffix", prompts_embedding[:, 1 + self.n_ctx :, :]
        )  # CLS and EOS

        self.dtype = clip_model.dtype

    # TODO: PromptLearner - Construct Prompts
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        pass

    # TODO: PromptLearner - Forward
    def forward(self, im_features):
        pass


# TODO: CustomCLIP
class CustomCLIP(nn.Module):
    def __init__(self, cfg, class_names, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, class_names, clip_model)
        
        
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
