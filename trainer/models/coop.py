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

        with torch.no_grad():
            prompts_embedding = clip_model.token_embedding(self.prompts_tokenized).type(
                clip_model.dtype
            )

        self.register_buffer("token_prefix", prompts_embedding[:, :1, :])  # SOS
        self.register_buffer(
            "token_suffix", prompts_embedding[:, 1 + self.n_ctx :, :]
        )  # CLS and EOS

        self.class_token_position = cfg.MODEL.CoOp.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx

        # Copy ctx to every class
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        else:
            raise ValueError

        return prompts


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
        image_features = self.image_encoder(image)

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.promptes_tokenized)

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
