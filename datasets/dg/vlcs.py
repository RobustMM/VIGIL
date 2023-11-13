import glob
import os

from datasets.base_dataset import DatasetBase, Datum
from datasets.build_dataset import DATASET_REGISTRY
from utils import listdir_nonhidden


@DATASET_REGISTRY.register()
class VLCS(DatasetBase):
    """
    VLCS Statistics:
        - 4 domains: CALTECH, LABELME, PASCAL, SUN.
        - 5 categories: bird, car, chair, dog, and person.

    Reference:
        - Torralba and Efros. Unbiased look at dataset bias. CVPR 2011.
    """

    def __init__(self, cfg):
        self._dataset_dir = "vlcs"
        self._domains = ["caltech", "labelme", "pascal", "sun"]
        self._data_url = (
            "https://drive.google.com/uc?id=1o8RlEMyT5D3Pmcw0YbyGQA7uDzPY4xbi"
        )
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self._dataset_dir = os.path.join(root, self.dataset_dir)

        if not os.path.exists(self._dataset_dir):
            self.download_data_from_gdrive(os.path.join(root, "vlcs.zip"))

        self.check_input_domains(cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS)

        train_data = self.read_data(cfg.DATASET.SOURCE_DOMAINS, "train")
        val_data = self.read_data(cfg.DATASET.SOURCE_DOMAINS, "crossval")
        test_data = self.read_data(cfg.DATASET.TARGET_DOMAINS, "test")

        super().__init__(
            dataset_dir=self._dataset_dir,
            domains=self._domains,
            data_url=self._data_url,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
        )

    def read_data(self, input_domains, split):
        def _load_data_from_directory(directory):
            folder_names = listdir_nonhidden(directory)
            images_ = []

            for class_label, folder_name in enumerate(folder_names):
                img_paths = glob.glob(os.path.join(directory, folder_name, "*"))

                for img_path in img_paths:
                    images_.append((img_path, class_label))

            return images_

        img_datums = []

        for domain_label, domain_name in enumerate(input_domains):
            split_dir = os.path.join(self._dataset_dir, domain_name, split)
            img_path_class_label_list = _load_data_from_directory(split_dir)

            for img_path, class_label in img_path_class_label_list:
                class_name = img_path.split("/")[-2].lower()

                img_datum = Datum(
                    img_path=img_path,
                    class_label=class_label,
                    domain_label=domain_label,
                    class_name=class_name,
                )
                img_datums.append(img_datum)

        return img_datums
