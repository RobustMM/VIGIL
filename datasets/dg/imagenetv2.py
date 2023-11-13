import glob
import os

from datasets.base_dataset import DatasetBase, Datum
from datasets.build_dataset import DATASET_REGISTRY

from .imagenet import ImageNet


@DATASET_REGISTRY.register()
class ImageNetV2(DatasetBase):
    """ImageNetV2

    This dataset is used for testing only.
    """

    def __init__(self, cfg):
        self._dataset_dir = "imagenetv2"
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self._dataset_dir = os.path.join(root, self._dataset_dir)
        text_file = os.path.join(self._dataset_dir, "classnames.txt")
        class_names_labels = ImageNet.read_class_names_labels(text_file)

        data = self.read_data(class_names_labels)

        super().__init__(
            dataset_dir=self._dataset_dir,
            domains=0,
            train_data=data,
            val_data=data,
            test_data=data,
        )

    def read_data(self, class_names_labels):
        img_dir = os.path.join(
            self._dataset_dir, "imagenetv2-matched-frequency-format-val"
        )
        folder_names = list(class_names_labels.keys())
        img_datums = []

        for class_label in range(1000):
            folder_name = folder_names[class_label]
            class_name, _ = class_names_labels[folder_name]

            img_paths = glob.glob(os.path.join(img_dir, str(class_label), "*"))

            for img_path in img_paths:
                img_datum = Datum(
                    img_path=img_path,
                    class_label=class_label,
                    domain_label=0,
                    class_name=class_name,
                )
                img_datums.append(img_datum)
        return img_datums
