from trainer import MODEL_REGISTERY, Trainer


@MODEL_REGISTERY.register()
class ClipModel(Trainer):

    def build_model(self):
        print("Build Model")
        self.model = "CLIP"

    def test(self):
        print("Strat Evaluation")
