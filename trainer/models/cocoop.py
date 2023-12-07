import os
from collections import OrderedDict

import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer

from optim import build_lr_scheduler, build_optimizer
from trainer import MODEL_REGISTRY, Trainer

_tokenizer = SimpleTokenizer()


class CustomTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, prompts_tokenized):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), prompts_tokenized.argmax(dim=-1)]
            @ self.text_projection
        )

        return x


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


class CustomCLIP(nn.Module):
    def __init__(self, cfg, class_names, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, class_names, clip_model)
        self.promptes_tokenized = self.prompt_learner.prompts_tokenized
        self.image_encoder = clip_model.visual
        self.text_encoder = CustomTextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    # TODO: CustomCLIP - Forward
    def forward(self, image, label=None):
        pass


@MODEL_REGISTRY.register()
class CoCoOp(Trainer):
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

        print("Turning Off Gradients in Image and Text Encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        # Double check
        enabled_params = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled_params.add(name)
        print("Parameters to be updated: {}".format(enabled_params))

        self.model.to(self.device)

        # NOTE: Only Give prompt_learner to the Optimizer
        self.optimizer = build_optimizer(self.model.prompt_learner, self.cfg.OPTIM)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, self.cfg.OPTIM)

        self.model_registeration(
            "prompt_learner",
            self.model.prompt_learner,
            self.optimizer,
            self.lr_scheduler,
        )

    # TODO: CoCoOp - Forward Backward
    def forward_backward(self, batch_data):
        pass
