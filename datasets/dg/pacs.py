import os

from datasets.base_dataset import DatasetBase, Datum
from datasets.build_dataset import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PACS(DatasetBase):
    """
    PACS Statistics:
        - 4 domains: Photo (1,670), Art (2,048), Cartoon (2,344), Sketch (3,929).
        - 7 categories: dog, elephant, giraffe, guitar, horse, house and person.

    Reference:
        - Li et al. Deeper, broader and artier domain generalization. ICCV 2017.
    """

    def __init__(self, cfg):
        self._dataset_dir = "pacs"
        self._domains = ["art_painting", "cartoon", "photo", "sketch"]
        self._data_url = (
            "https://drive.google.com/uc?id=1wN5jJiG3makr8D2iDX5CI7oFXGua8nYB"
        )
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self._dataset_dir = os.path.join(root, self.dataset_dir)
        self._image_dir = os.path.join(self._dataset_dir, "images")
        self._split_dir = os.path.join(self._dataset_dir, "splits")
        # The following images contain errors and should be ignored
        self._error_img_paths = ["sketch/dog/n02103406_4068-1.png"]

        if not os.path.exists(self._dataset_dir):
            self.download_data_from_gdrive(os.path.join(root, "pacs.zip"))

        self.check_input_domains(cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS)

        train_data = self.read_data(cfg.DATASET.SOURCE_DOMAINS, "train")
        val_data = self.read_data(cfg.DATASET.SOURCE_DOMAINS, "crossval")
        test_data = self.read_data(cfg.DATASET.TARGET_DOMAINS, "all")

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
            images_ = []
            with open(directory, "r") as file:
                lines = file.readlines()
                for line in lines:
                    img_path, class_label = line.split(" ")
                    if img_path in self._error_img_paths:
                        continue
                    img_path = os.path.join(self._image_dir, img_path)
                    class_label = int(class_label) - 1
                    images_.append((img_path, class_label))

            return images_

        img_datums = []

        for domain_label, domain_name in enumerate(input_domains):
            if split == "all":
                train_dir = os.path.join(
                    self._split_dir, domain_name + "_train_kfold.txt"
                )
                img_path_class_label_list = _load_data_from_directory(train_dir)
                val_dir = os.path.join(
                    self._split_dir, domain_name + "_crossval_kfold.txt"
                )
                img_path_class_label_list += _load_data_from_directory(val_dir)
            else:
                split_dir = os.path.join(
                    self._split_dir, domain_name + "_" + split + "_kfold.txt"
                )
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
