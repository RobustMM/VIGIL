import os

import gdown
import zipfile


class Datum:
    def __init__(self, img_path, class_label, domain_label, class_name):
        """Data instance which defines the basic attributes.

        Args:
            img_path (str): Image path
            class_label (int): Class label
            domain_label (int): Domain label
            class_name (str): Class name
        """
        assert isinstance(img_path, str)
        assert os.path.isfile(img_path)

        self._img_path = img_path
        self._class_label = class_label
        self._domain_label = domain_label
        self._class_name = class_name

    @property
    def img_path(self):
        return self._img_path

    @property
    def class_label(self):
        return self._class_label

    @property
    def domain_label(self):
        return self._domain_label

    @property
    def class_name(self):
        return self._class_name


class DatasetBase:
    def __init__(
        self,
        dataset_dir,
        domains,
        data_url=None,
        train_data=None,
        val_data=None,
        test_data=None,
    ):
        self._dataset_dir = dataset_dir
        self._domains = domains
        self._data_url = data_url
        self._train_data = train_data
        self._val_data = val_data
        self._test_data = test_data
        self._num_classes = self.get_num_classes()
        (
            self._class_label_name_mapping,
            self._class_names,
        ) = self.get_class_label_name_mapping()

    @property
    def dataset_dir(self):
        return self._dataset_dir

    @property
    def domains(self):
        return self._domains

    @property
    def data_url(self):
        return self._data_url

    @property
    def train_data(self):
        return self._train_data

    @property
    def val_data(self):
        return self._val_data

    @property
    def test_data(self):
        return self._test_data

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def class_label_name_mapping(self):
        return self._class_label_name_mapping

    @property
    def class_names(self):
        return self._class_names

    def get_num_classes(self):
        label_set = set()
        for datum in self._train_data:
            label_set.add(datum.class_label)

        return max(label_set) + 1

    def get_class_label_name_mapping(self):
        container = set()
        for datum in self._train_data:
            container.add((datum.class_label, datum.class_name))
        class_label_name_mapping = {
            class_label: class_name for class_label, class_name in container
        }
        class_labels = list(class_label_name_mapping.keys())
        class_labels.sort()
        class_names = [
            class_label_name_mapping[class_label] for class_label in class_labels
        ]

        return class_label_name_mapping, class_names

    def check_input_domains(self, source_domains, target_domain):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domain)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self._domains:
                raise ValueError(
                    "Input Domain Must Belong to {}, " "but Got [{}]".format(
                        self._domains, domain
                    )
                )

    def download_data_from_gdrive(self, dst):
        gdown.download(self._data_url, dst, quiet=False)

        zip_ref = zipfile.ZipFile(dst, "r")
        zip_ref.extractall(os.path.dirname(dst))
        zip_ref.close()
        print("File Extracted to {}".format(os.path.dirname(dst)))
        os.remove(dst)
