import glob
import os

from datasets.base_dataset import DatasetBase, Datum
from datasets.build_dataset import DATASET_REGISTRY
from utils import listdir_nonhidden


@DATASET_REGISTRY.register()
class TerraInc(DatasetBase):
    """
    TerraIncognita Statistics:
        - A dataset consisting of wild animal photographs.
        - 4 domains based on the location where the images were captured: L100, L38, L43, L46.
        - 24,330 images.
        - 10 categories.
        - https://lila.science/datasets/caltech-camera-traps

    Reference:
        - Sara et al. Recognition in Terra Incognita. ECCV 2018.
    """

    def __init__(self, cfg):
        self._dataset_dir = "terra_incognita"
        self._domains = ["location_38", "location_43", "location_46", "location_100"]
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self._dataset_dir = os.path.join(root, self.dataset_dir)

        self.check_input_domains(cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS)

        train_data = self.read_data(cfg.DATASET.SOURCE_DOMAINS)
        val_data = self.read_data(cfg.DATASET.TARGET_DOMAINS)
        test_data = self.read_data(cfg.DATASET.TARGET_DOMAINS)

        super().__init__(
            dataset_dir=self._dataset_dir,
            domains=self._domains,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
        )

    def read_data(self, input_domains):
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
            img_path_class_label_list = _load_data_from_directory(
                os.path.join(self._dataset_dir, domain_name)
            )

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
