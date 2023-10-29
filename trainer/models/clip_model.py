from trainer import MODEL_REGISTERY, Trainer


@MODEL_REGISTERY.register()
class ClipModel(Trainer):

    def build_model(self):
        print("Build Model")

    def forward_backward(self, batch_data):
        print("Forward")
