import clip

from trainer import MODEL_REGISTERY, Trainer


@MODEL_REGISTERY.register()
class ClipModel(Trainer):

    def build_model(self):
        self.model, self.preprocess = clip.load(self.cfg.MODEL.BACKBONE, device="cuda", jit=False)
        self.model.eval()
        self.model.to(self.device)

    def test(self):
        print("Strat Evaluation")
