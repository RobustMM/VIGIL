import glob
import json
import os

from datasets.base_dataset import DatasetBase, Datum
from datasets.build_dataset import DATASET_REGISTRY
from utils import listdir_nonhidden


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

        super().__init__(
            dataset_dir=self._dataset_dir,
            domains=0,
            train_data=data,
            val_data=data,
            test_data=data,
        )

    def read_data(self):
        with open(
            os.path.join(self._dataset_dir, "folder_to_objectnet_label.json"), "r"
        ) as file:
            folder_class_name_mapping = json.load(file)

        image_dir = os.path.join(self._dataset_dir, "images")
        folder_names = listdir_nonhidden(image_dir)
        img_datums = []

        for class_label, folder_name in enumerate(folder_names):
            class_name = folder_class_name_mapping[folder_name]

            img_paths = glob.glob(os.path.join(image_dir, folder_name, "*.png"))

            for img_path in img_paths:
                img_datum = Datum(
                    img_path=img_path,
                    class_label=class_label,
                    domain_label=0,
                    class_name=class_name,
                )
                img_datums.append(img_datum)

        return img_datums
