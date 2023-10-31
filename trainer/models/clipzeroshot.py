from trainer import Trainer, MODEL_REGISTERY


@MODEL_REGISTERY.register()
class CLIPZeroshot(Trainer):
    def build_model(self):
        print("Hi")
        exit()
