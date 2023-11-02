import os.path as osp


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
        assert osp.isfile(img_path)

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
        print("Hi")
        exit()

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

    def get_num_classes(self):
        label_set = set()
        for datum in self._train_data:
            label_set.add(datum.class_label)

        return max(label_set) + 1
