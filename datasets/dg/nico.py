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
        print("Hi")
        exit()
