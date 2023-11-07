# import clip
from clip import clip

from trainer import MODEL_REGISTERY, Trainer

PROMPT_TEMPLATES = {"ImageNet": "a photo of a {}."}


@MODEL_REGISTERY.register()
class CLIPZeroshot(Trainer):
    def build_model(self):
        class_names = self.data_manager.dataset.class_names

        clip_model, preprocess = clip.load(
            self.cfg.MODEL.BACKBONE,
            device=self.device,
            download_root="/data/dzha866/Project/VIGIL/data/",
        )
        prompt_template = PROMPT_TEMPLATES[self.cfg.DATASET.NAME]
        print(prompt_template)

        exit()
