import glob
import os
import xml.etree.ElementTree as ET
from collections import OrderedDict

from datasets.base_dataset import DatasetBase, Datum
from datasets.build_dataset import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ImageNet(DatasetBase):
    def __init__(self, cfg):
        self._dataset_dir = "imagenet"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self._dataset_dir = os.path.join(root, self.dataset_dir)

        text_file = os.path.join(self._dataset_dir, "classnames.txt")
        class_names_labels = self.read_class_names_labels(text_file)
        # TODO: Debug Train_Data_Order
        train_data = self.read_data(class_names_labels, "val")
        test_data = self.read_data(class_names_labels, "val")

        super().__init__(
            dataset_dir=self._dataset_dir,
            domains=0,
            train_data=train_data,
            val_data=test_data,
            test_data=test_data,
        )

    @staticmethod
    def read_class_names_labels(text_file):
        """
        Args:
            text_file (str): Path of file that contains all folders' names and corresponding class names

        Returns:
            OrderedDict: Key-value pairs of <folder name>: <class name>
        """
        class_names_labels = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            class_label = 0
            for line in lines:
                line = line.strip().split(" ")
                folder_name = line[0]
                class_name = " ".join(line[1:])
                class_names_labels[folder_name] = (class_name, class_label)
                class_label += 1

        return class_names_labels

    def read_data(self, class_names_labels, split_dir):
        if split_dir == "train":
            return self._read_data_train(class_names_labels, split_dir)
        elif split_dir == "val" or split_dir == "test":
            return self._read_data_test(class_names_labels, split_dir)

    def _read_data_train(self, class_names_labels, split_dir):
        split_dir = os.path.join(self._dataset_dir, split_dir)
        folder_names = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        img_datums = []

        for folder_name in folder_names:
            class_name, class_label = class_names_labels[folder_name]

            img_paths = glob.glob(os.path.join(split_dir, folder_name, "*"))
            for img_path in img_paths:
                img_datum = Datum(
                    img_path=img_path,
                    class_label=class_label,
                    domain_label=0,
                    class_name=class_name,
                )
                img_datums.append(img_datum)

        return img_datums

    def _read_data_test(self, class_names_labels, split_dir):
        split_dir = os.path.join(self._dataset_dir, split_dir)

        img_paths = sorted(glob.glob(os.path.join(split_dir, "*.JPEG")))

        img_datums = []

        for img_path in img_paths:
            annotation_path = (
                split_dir
                + "_annotations/"
                + img_path.split(".")[0].split("/")[-1]
                + ".xml"
            )
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            class_name, class_label = class_names_labels[root.find(".//name").text]
            img_datum = Datum(
                img_path=img_path,
                class_label=class_label,
                domain_label=0,
                class_name=class_name,
            )
            img_datums.append(img_datum)

        return img_datums
