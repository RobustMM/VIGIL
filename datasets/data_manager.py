import torch
from PIL import Image
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset

from .build_dataset import build_dataset
from .samplers import build_sampler
from .transforms import build_transform


def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    transform=None,
    is_train=True,
):
    sampler = build_sampler(sampler_type=sampler_type, data_source=data_source)

    data_loader = torch.utils.data.DataLoader(
        dataset=DatasetWrapper(cfg, data_source, transform, is_train),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train,
        pin_memory=torch.cuda.is_available(),
    )

    assert len(data_loader) > 0

    return data_loader


class DataManager:
    def __init__(self, cfg):
        self.dataset = build_dataset(cfg)

        transform_train = build_transform(cfg, is_train=True)
        transform_test = build_transform(cfg, is_train=False)

        self.data_loader_train = build_data_loader(
            cfg=cfg,
            sampler_type=cfg.DATALOADER.TRAIN.SAMPLER,
            data_source=self.dataset.train_data,
            batch_size=cfg.DATALOADER.TRAIN.BATCH_SIZE,
            transform=transform_train,
            is_train=True,
        )

        self.data_loader_val = None
        if self.dataset.val_data:
            self.data_loader_val = build_data_loader(
                cfg=cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=self.dataset.val_data,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                transform=transform_test,
                is_train=False,
            )

        self.data_loader_test = build_data_loader(
            cfg=cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=self.dataset.test_data,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            transform=transform_test,
            is_train=False,
        )

        self._num_classes = self.dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._class_label_name_mapping = self.dataset.class_label_name_mapping

        self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def class_label_name_mapping(self):
        return self._class_label_name_mapping

    def show_dataset_summary(self, cfg):
        table = [["Dataset", cfg.DATASET.NAME]]
        if cfg.DATASET.SOURCE_DOMAINS:
            table.append(["Source Domains", cfg.DATASET.SOURCE_DOMAINS])
        if cfg.DATASET.TARGET_DOMAINS:
            table.append(["Target Domains", cfg.DATASET.TARGET_DOMAINS])
        table.append(["# Classes", f"{self.num_classes:,}"])
        table.append(["# Train Data", f"{len(self.dataset.train_data):,}"])
        if self.dataset.val_data:
            table.append(["# Val Data", f"{len(self.dataset.val_data):,}"])
        table.append(["# Test Data", f"{len(self.dataset.test_data):,}"])

        print(tabulate(table))


class DatasetWrapper(TorchDataset):
    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform
        self.in_train = is_train

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        datum = self.data_source[idx]

        output = {
            "img_path": datum.img_path,
            "domain_label": datum.domain_label,
            "class_label": datum.class_label,
            "index": idx,
        }

        img = Image.open(datum.img_path).convert("RGB")
        output["img"] = self.transform(img)

        return output
