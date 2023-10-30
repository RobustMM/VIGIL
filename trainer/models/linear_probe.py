# import clip

from trainer import MODEL_REGISTERY, Trainer


@MODEL_REGISTERY.register()
class LinearProbe(Trainer):

    def build_model(self):
        self.model, self.preprocess = clip.load(self.cfg.MODEL.BACKBONE, device="cuda", jit=False,
                                                download_root=self.cfg.OUTPUTS)
        self.model.eval()
        self.model.to(self.device)

    def test(self):
        print("Strat Evaluation")
