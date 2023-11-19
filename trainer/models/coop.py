from trainer import MODEL_REGISTERY, Trainer

from clip import clip


@MODEL_REGISTERY.register()
class CoOp(Trainer):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        pass

    def build_model(self):
        # print(self.cfg)
        # print(self.data_manager.dataset.class_names)

        self.clip_model, _ = clip.load(
            self.cfg.MODEL.BACKBONE,
            device="cpu",
            download_root="/data/dzha866/Project/VIGIL/data/",
        )

        print(self.cfg.MODEL)

        exit()
