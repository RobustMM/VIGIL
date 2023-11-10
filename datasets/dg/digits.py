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
        self._data_url = os.path.join(root, self.dataset_dir)
