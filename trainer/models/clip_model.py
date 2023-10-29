from trainer import MODEL_REGISTERY, GenericTrainer


@MODEL_REGISTERY.register()
class ClipModel(GenericTrainer):

    def build_model(self):
        print("Build Model")

    def forward_backward(self, batch_data):
        print("Forward")
