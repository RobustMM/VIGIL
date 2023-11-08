import os
import pickle
from collections import OrderedDict

from datasets.base_dataset import DatasetBase, Datum
from datasets.build_dataset import DATASET_REGISTRY
from utils import listdir_nonhidden

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class ImageNet(DatasetBase):
    def __init__(self, cfg):
        self._dataset_dir = "imagenet"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self._dataset_dir = os.path.join(root, self.dataset_dir)
        self._preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")

        if os.path.exists(self._preprocessed):
            with open(self._preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train_data = preprocessed["train_data"]
                test_data = preprocessed["test_data"]
        else:
            text_file = os.path.join(self._dataset_dir, "classnames.txt")
            class_names = self.read_class_names(text_file)
            train_data = self.read_data(class_names, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test_data = self.read_data(class_names, "val")

            preprocessed = {"train_data": train_data, "test_data": test_data}
            with open(self._preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        # TODO: May need to refactor to base class
        train_data, test_data = OxfordPets.subsample_classes(
            train_data, test_data, subsample=subsample
        )

        super().__init__(
            dataset_dir=self._dataset_dir,
            domains=0,
            train_data=train_data,
            val_data=test_data,
            test_data=test_data,
        )

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
        if split_dir == "train":
            self._read_data_train(class_names, split_dir)
        elif split_dir == "val" or split_dir == "test":
            self._read_data_test(class_names, split_dir)

    def _read_data_train(self, class_names, split_dir):
        split_dir = os.path.join(self._dataset_dir, split_dir)
        folder_names = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        img_datums = []

        for class_label, folder_name in enumerate(folder_names):
            img_names = listdir_nonhidden(os.path.join(split_dir, folder_name))
            class_name = class_names[folder_name]
            for img_name in img_names:
                img_path = os.path.join(split_dir, folder_name, img_name)
                img_datum = Datum(
                    img_path=img_path,
                    class_label=class_label,
                    domain_label=0,
                    class_name=class_name,
                )
                img_datums.append(img_datum)

        return img_datums

    def _read_data_test(self, class_names, split_dir):
        print("Read Test Data")
        exit()
