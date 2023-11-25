import os

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
        print("CustomTextEncoder Forward")
        exit()


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

    # TODO: PromptLearner Forward
    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        exit()


class CustomCLIP(nn.Module):
    def __init__(self, cfg, class_names, clip_model):
        super().__init__()

        self.prompt_learner = PromptLearner(cfg, class_names, clip_model)
        self.promptes_tokenized = self.prompt_learner.prompts_tokenized
        self.image_encoder = clip_model.visual
        self.text_encoder = CustomTextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    # TODO: CustomCLIP
    def forward(self, image):
        print("Custom CLIP Forward")
        image_embeddings = self.image_encoder(image)

        prompts = self.prompt_learner()
        print(prompts)

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

        print("Building Custom CLIP")
        self.model = CustomCLIP(
            self.cfg, self.data_manager.dataset.class_names, clip_model
        )

        print("Turning Off Gradients in Image and Text Encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        self.model.to(self.device)

        # NOTE: Only Give prompt_learner to the Optimizer
        self.optimizer = build_optimizer(self.model.prompt_learner, self.cfg.OPTIM)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, self.cfg.OPTIM)

    # TODO: CoOp Forward
    def forward_backward(self, batch_data):
        image, class_label = self.parse_batch_train(batch_data)
        output = self.model(image)
        print(output)

        exit()

    def parse_batch_train(self, batch_data):
        image = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)

        return image, class_label
