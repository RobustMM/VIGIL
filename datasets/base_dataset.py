class Datum:
    pass


class DatasetBase:
    def __init__(
        self, dataset_dir, domains, data_url=None, train_data=None, test_data=None
    ):
        self.dataset_dir = dataset_dir
        self.domains = domains
        self.data_url = data_url
        self.train_data = train_data
        self.test_data = test_data
