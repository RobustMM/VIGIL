from .build_dataset import build_dataset


class DataManager:
    def __init__(self, cfg):
        dataset = build_dataset(cfg)

        # TODO: Build Transform
        tfm_train = build_transform(cfg, is_train=True)
        tfm_test = build_transform(cfg, is_train=False)
