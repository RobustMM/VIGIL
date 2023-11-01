import os
import pickle

from datasets.base_dataset import DatasetBase
from datasets.build_dataset import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ImageNet(DatasetBase):
    def __init__(self, cfg):
        self.dataset_dir = "imagenet"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            print(text_file)
            exit()
            classnames = self.read_classnames(text_file)
            print(classnames)
            exit()
    
    @staticmethod
    def read_classnames(text_file):
        pass
