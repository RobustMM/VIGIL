import torch
from torch.utils.data import Dataset as TorchDataset
from PIL import Image

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

    print(len(data_loader.dataset))
    print(data_loader.batch_size)
    print(data_loader.sampler)
    print(data_loader.num_workers)
    print(data_loader.drop_last)
    print(data_loader.pin_memory)

    assert len(data_loader) > 0

    return data_loader


class DataManager:
    def __init__(self, cfg):
        dataset = build_dataset(cfg)

        transform_train = build_transform(cfg, is_train=True)
        transform_test = build_transform(cfg, is_train=False)

        data_loader_train = build_data_loader(
            cfg=cfg,
            sampler_type=cfg.DATALOADER.TRAIN.SAMPLER,
            data_source=dataset.train_data,
            batch_size=cfg.DATALOADER.TRAIN.BATCH_SIZE,
            transform=transform_train,
            is_train=True,
        )

        val_loader = None
        if dataset.val_data:
            val_loader = build_data_loader(
                cfg=cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val_data,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                transform=transform_test,
                is_train=False,
            )

        test_loader = build_data_loader(
            cfg=cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test_data,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            transform=transform_test,
            is_train=False,
        )


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
