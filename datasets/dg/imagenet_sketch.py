import os

from datasets.base_dataset import DatasetBase, Datum
from datasets.build_dataset import DATASET_REGISTRY
from utils import listdir_nonhidden

from .imagenet import ImageNet


@DATASET_REGISTRY.register()
class ImageNetSketch(DatasetBase):
    """ImageNet-Sketch

    This dataset is used for testing only.
    """

    def __init__(self, cfg):
        self._dataset_dir = "imagenet-sketch"
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
        image_dir = os.path.join(self._dataset_dir, "images")
        folder_names = listdir_nonhidden(image_dir)
        img_datums = []

        for class_label, folder_name in enumerate(folder_names):
            img_names = listdir_nonhidden(os.path.join(image_dir, folder_name))
            class_name, _ = class_names_labels[folder_name]

            for img_name in img_names:
                img_path = os.path.join(image_dir, folder_name, img_name)
                img_datum = Datum(
                    img_path=img_path,
                    class_label=class_label,
                    domain_label=0,
                    class_name=class_name,
                )
                img_datums.append(img_datum)

        return img_datums
