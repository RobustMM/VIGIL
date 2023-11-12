import torch
from clip import clip

from trainer import MODEL_REGISTERY, Trainer

PROMPT_TEMPLATES = {
    "ImageNet": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "Digits": "a picture of a {}.",
    "PACS": "a picture of a {}.",
    "OfficeHome": "a picture of a {}.",
    "VLCS": "a picture of a {}.",
    "NICO": "a picture of a {}.",
}


@MODEL_REGISTERY.register()
class CLIPZeroshot(Trainer):
    def build_model(self):
        class_names = self.data_manager.dataset.class_names

        self.clip_model, preprocess = clip.load(
            self.cfg.MODEL.BACKBONE,
            device=self.device,
            download_root="/data/dzha866/Project/VIGIL/data/",
        )
        prompt_template = PROMPT_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [
            prompt_template.format(class_name.replace("_", " "))
            for class_name in class_names
        ]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(prompt) for prompt in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(prompts)
            self.text_features = self.text_features / self.text_features.norm(
                dim=-1, keepdim=True
            )

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits
