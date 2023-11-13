import os

from datasets.base_dataset import DatasetBase, Datum
from datasets.build_dataset import DATASET_REGISTRY
from utils import listdir_nonhidden

from .imagenet import ImageNet


@DATASET_REGISTRY.register()
class ObjectNet(DatasetBase):
    """ObjectNet

    This dataset is used for testing only.
    """

    def __init__(self, cfg):
        self._dataset_dir = "objectnet"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self._dataset_dir = os.path.join(root, self._dataset_dir)

        data = self.read_data()
        
    