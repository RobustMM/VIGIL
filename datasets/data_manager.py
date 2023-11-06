from .build_dataset import build_dataset
from .transforms import build_transform

from torch.utils.data import Dataset as TorchDataset


def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    transform=None,
    is_train=True,
    dataset_wrapper=None,
):
    pass


class DataManager:
    def __init__(self, cfg):
        dataset = build_dataset(cfg)

        transform_train = build_transform(cfg, is_train=True)
        transform_test = build_transform(cfg, is_train=False)

        # TODO: Build Data Loader - Train
        print(cfg.DATALOADER)
        data_loader_train = build_data_loader(
            cfg=cfg,
            sampler_type=cfg.DATALOADER.TRAIN.SAMPLER,
            data_source=dataset.train_data,
            batch_size=cfg.DATALOADER.TRAIN.BATCH_SIZE,
            transform=transform_train,
            is_train=True,
        )


class DatasetWrapper(TorchDataset):
    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform
        self.in_train = is_train

    def __len__(self):
        return len(self.data_source)
