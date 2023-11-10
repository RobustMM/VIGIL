import glob
import os
import sys

from datasets.base_dataset import DatasetBase, Datum
from datasets.build_dataset import DATASET_REGISTRY
from utils import listdir_nonhidden


@DATASET_REGISTRY.register()
class Digits(DatasetBase):
    """
    Digits contains 4 digit datasets:
        - MNIST: hand-written digits.
        - MNIST-M: variant of MNIST with blended background.
        - SVHN: street view house number.
        - SYN: synthetic digits.

    Reference:
        - Lecun et al. Gradient-based learning applied to document recognition. IEEE 1998.
        - Ganin et al. Domain-adversarial training of neural networks. JMLR 2016.
        - Netzer et al. Reading digits in natural images with unsupervised feature learning. NIPS-W 2011.
        - Zhou et al. Deep Domain-Adversarial Image Generation for Domain Generalisation. AAAI 2020.
    """

    def __init__(self, cfg):
        self._dataset_dir = "digits"
        self._domains = ["mnist", "mnist_m", "svhn", "syn"]
        self._data_url = (
            "https://drive.google.com/uc?id=1GK4B94SGABgOH0pguTxFQLO9tQVamsnz"
        )
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self._dataset_dir = os.path.join(root, self.dataset_dir)
        self._domain_info = {}

        if not os.path.exists(self._dataset_dir):
            self.download_data_from_gdrive(os.path.join(root, "digits.zip"))

        self.check_input_domains(cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS)

        exit()
        train_data = self.read_data(cfg.DATASET.SOURCE_DOMAINS, "train")
        val_data = self.read_data(cfg.DATASET.SOURCE_DOMAINS, "val")
        test_data = self.read_data(cfg.DATASET.TARGET_DOMAINS, "test")
