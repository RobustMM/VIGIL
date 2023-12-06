from trainer import MODEL_REGISTRY, Trainer


@MODEL_REGISTRY.register()
class CoCoOp(Trainer):
    # TODO: CoCoOp - Build_Model
    def build_model(self):
        print("Hi")
        exit()
