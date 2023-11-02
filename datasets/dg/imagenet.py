import os
import pickle
from collections import OrderedDict

from datasets.base_dataset import DatasetBase, Datum
from datasets.build_dataset import DATASET_REGISTRY
from utils import listdir_nonhidden


@DATASET_REGISTRY.register()
class ImageNet(DatasetBase):
    def __init__(self, cfg):
        self._dataset_dir = "imagenet"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self._dataset_dir = os.path.join(root, self.dataset_dir)
        self._image_dir = os.path.join(self.dataset_dir, "images")
        self._preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")

        if os.path.exists(self._preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self._dataset_dir, "classnames.txt")
            class_names = self.read_class_names(text_file)
            train = self.read_data(class_names, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_data(class_names, "val")
            exit()

    @staticmethod
    def read_class_names(text_file):
        """
        Args:
            text_file (str): Path of file that contains all folders' names and corresponding class names

        Returns:
            OrderedDict: Key-value pairs of <folder name>: <class name>
        """
        class_names = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder_name = line[0]
                class_name = " ".join(line[1:])
                class_names[folder_name] = class_name

        return class_names

    def read_data(self, class_names, split_dir):
        split_dir = os.path.join(self._image_dir, split_dir)
        folder_names = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        img_datums = []

        for class_label, folder_name in enumerate(folder_names):
            img_names = listdir_nonhidden(os.path.join(split_dir, folder_name))
            class_name = class_names[folder_name]
            for img_name in img_names:
                img_path = os.path.join(split_dir, folder_name, img_name)
                img_datum = Datum(img_path=img_path, class_label=class_label, domain_label=0, class_name=class_name)
