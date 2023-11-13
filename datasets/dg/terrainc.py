import os

from datasets.base_dataset import DatasetBase, Datum
from datasets.build_dataset import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class TerraInc(DatasetBase):
    """
    TerraIncognita Statistics:
        - 4 domains based on the location where the images were captured: L100, L38, L43, L46
        - 24,788 images.
        - 10 categories.
        - https://lila.science/datasets/caltech-camera-traps

    Reference:
        - Sara et al. Recognition in Terra Incognita. ECCV 2018.
    """

    def __init__(self, cfg):
        print("TerraInc")
        exit()
