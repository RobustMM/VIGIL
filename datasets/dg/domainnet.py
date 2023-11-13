import os

from datasets.base_dataset import DatasetBase, Datum
from datasets.build_dataset import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class DomainNet(DatasetBase):
    """
    DomainNet Statistics:
        - 6 distinct domains: Clipart, Infograph, Painting, Quickdraw, Real, Sketch.
        - Around 0.6M images.
        - 345 categories.
        - URL: http://ai.bu.edu/M3SDA/.

    Special note: the t-shirt class (327) is missing in painting_train.txt.

    Reference:
        - Peng et al. Moment Matching for Multi-Source Domain Adaptation. ICCV 2019.
    """

    def __init__(self, cfg):
        self._dataset_dir = "domainnet"
        self._domains = [
            "clipart",
            "infograph",
            "painting",
            "quickdraw",
            "real",
            "sketch",
        ]
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self._dataset_dir = os.path.join(root, self.dataset_dir)

        self.check_input_domains(cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS)

        train_data = self.read_data(cfg.DATASET.SOURCE_DOMAINS, "train")
        val_data = self.read_data(cfg.DATASET.SOURCE_DOMAINS, "test")
        test_data = self.read_data(cfg.DATASET.TARGET_DOMAINS, "all")

        super().__init__(
            dataset_dir=self._dataset_dir,
            domains=self._domains,
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
                    img_path = os.path.join(self._dataset_dir, img_path)
                    class_label = int(class_label)
                    images_.append((img_path, class_label))

                return images_

        img_datums = []

        for domain_label, domain_name in enumerate(input_domains):
            if split == "all":
                train_dir = os.path.join(self._dataset_dir, domain_name + "_train.txt")
                img_path_class_label_list = _load_data_from_directory(train_dir)
                test_dir = os.path.join(self._dataset_dir, domain_name + "_test.txt")
                img_path_class_label_list += _load_data_from_directory(test_dir)
            else:
                split_dir = os.path.join(
                    self._dataset_dir, domain_name + "_" + split + ".txt"
                )
                img_path_class_label_list = _load_data_from_directory(split_dir)

            for img_path, class_label in img_path_class_label_list:
                class_name = str(img_path.split("/")[-2].lower())

                img_datum = Datum(
                    img_path=img_path,
                    class_label=class_label,
                    domain_label=domain_label,
                    class_name=class_name,
                )
                img_datums.append(img_datum)

        return img_datums
