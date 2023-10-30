from utils import Registry, check_availability

DATASET_REGISTRY = Registry("DATASET")


def build_dataset(cfg):
    available_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.NAME, available_datasets)
    print("Build Dataset: {}".format(cfg.DATASET.NAME))

    return DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg)
