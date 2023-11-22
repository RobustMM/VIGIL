from utils import Registry, check_availability

MODEL_REGISTRY = Registry("MODEL")


def build_trainer(cfg):
    available_models = MODEL_REGISTRY.registered_names()
    check_availability(cfg.MODEL.NAME, available_models)
    print("Build Trainer: {}".format(cfg.MODEL.NAME))

    return MODEL_REGISTRY.get(cfg.MODEL.NAME)(cfg)
