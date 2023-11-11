import glob
import os

from datasets.base_dataset import DatasetBase, Datum
from datasets.build_dataset import DATASET_REGISTRY
from utils import listdir_nonhidden


@DATASET_REGISTRY.register()
class Digits(DatasetBase):
    """
    Digits contains 4 digit datasets:
        - MNIST: hand-written digits.
        - MNIST-M: variant of MNIST with blended background.
        - SVHN: street view house number.
        - SYN: synthetic digits.

    Reference:
        - Lecun et al. Gradient-based learning applied to document recognition. IEEE 1998.
        - Ganin et al. Domain-adversarial training of neural networks. JMLR 2016.
        - Netzer et al. Reading digits in natural images with unsupervised feature learning. NIPS-W 2011.
        - Zhou et al. Deep Domain-Adversarial Image Generation for Domain Generalisation. AAAI 2020.
    """

    def __init__(self, cfg):
        self._dataset_dir = "digits"
        self._domains = ["mnist", "mnist_m", "svhn", "syn"]
        self._data_url = (
            "https://drive.google.com/uc?id=1GK4B94SGABgOH0pguTxFQLO9tQVamsnz"
        )
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self._dataset_dir = os.path.join(root, self.dataset_dir)

        if not os.path.exists(self._dataset_dir):
            self.download_data_from_gdrive(os.path.join(root, "digits.zip"))

        self.check_input_domains(cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS)

        train_data = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "train")
        val_data = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "val")
        test_data = self._read_data(cfg.DATASET.TARGET_DOMAINS, "all")

        super().__init__(
            dataset_dir=self._dataset_dir,
            domains=self._domains,
            data_url=self._data_url,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
        )

    def _read_data(self, input_domains, split):
        def _load_data_from_directory(directory):
            folder_names = listdir_nonhidden(directory)
            images_ = []

            for class_label, folder_name in enumerate(folder_names):
                img_paths = glob.glob(os.path.join(directory, folder_name, "*.jpg"))

                for img_path in img_paths:
                    images_.append((img_path, class_label))

            return images_

        img_datums = []

        for domain_label, domain_name in enumerate(input_domains):
            if split == "all":
                train_dir = os.path.join(self._dataset_dir, domain_name, "train")
                img_path_class_label_list = _load_data_from_directory(train_dir)
                val_dir = os.path.join(self._dataset_dir, domain_name, "val")
                img_path_class_label_list += _load_data_from_directory(val_dir)
            else:
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
