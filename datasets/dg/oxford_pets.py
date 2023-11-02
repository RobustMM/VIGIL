from datasets.base_dataset import DatasetBase
from datasets.build_dataset import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class OxfordPets(DatasetBase):
    def __init__(self, cfg):
        self._dataset_dir = "oxford_pets"

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args
