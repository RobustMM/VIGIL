from utils import Registry, check_availability

MODEL_REGISTERY = Registry("MODEL")


def build_trainer(cfg):
    available_models = MODEL_REGISTERY.registered_names()
    check_availability(cfg.MODEL.NAME, available_models)