class Datum:
    pass


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
