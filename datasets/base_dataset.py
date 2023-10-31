class DatasetBase:
    def __init__(
        self, dataset_dir, domains, data_url=None, train_data=None, test_data=None
    ):
        self._dataset_dir = dataset_dir
        self._domains = domains
        self._data_url = data_url
        self._train_data = train_data
        self._test_data = test_data

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
    def test_data(self):
        return self._test_data
