import os

from datasets.base_dataset import DatasetBase, Datum
from datasets.build_dataset import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class NICO(DatasetBase):
    """
    NICO++ Statistics:
        - 88,866 images.
        - 6 domains: autumn, dim, grass, outdoor, rock, water
        - 60 categories.
        - url: https://nicochallenge.com/dataset

    Reference:
        - Xingxuan Zhang et al. NICO++: Towards Better Benchmarking for Domain Generalization. arXiv 2022.
    """

    def __init__(self, cfg):
        self._dataset_dir = "nico"
        self._domains = ["autumn", "dim", "grass", "outdoor", "rock", "water"]
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self._dataset_dir = os.path.join(root, self.dataset_dir)

        self.check_input_domains(cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS)

        train_data = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "train")
        exit()
        val_data = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "test")
        test_data = self._read_data(cfg.DATASET.TARGET_DOMAINS, "all")

        super().__init__(
            dataset_dir=self._dataset_dir,
            domains=self._domains,
            data_url=self._data_url,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
        )

    def _read_data(self, input_domains, split):
        def _load_data_from_directory(directory):
            images_ = []

            with open(directory, "r") as file:
                lines = file.readlines()
                for line in lines:
                    print(line)
                exit()

        img_datums = []

        for domain_label, domain_name in enumerate(input_domains):
            if split == "all":
                pass
            else:
                split_dir = os.path.join(
                    self._dataset_dir, domain_name + "_" + split + ".txt"
                )
                img_path_class_label_list = _load_data_from_directory(split_dir)
                print(len(img_path_class_label_list))

        exit()
