import torch
from clip import clip
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from trainer import MODEL_REGISTERY, Trainer


@MODEL_REGISTERY.register()
class CLIPLinearProbe(Trainer):
    def __init__(self, cfg):
        self.num_step = 8

        super().__init__(cfg=cfg)

    def build_model(self):
        print("Build CLIP LinearProbe")

        clip_model, preprocess = clip.load(
            self.cfg.MODEL.BACKBONE,
            device=self.device,
            download_root="/data/dzha866/Project/VIGIL/data/",
        )

        self.feature_train, self.class_label_train = self.get_feature(
            clip_model, self.data_loader_train
        )
        # self.feature_val, self.class_label_val = self.get_feature(
        #     clip_model, self.data_loader_val
        # )
        # self.feature_test, self.class_label_test = self.get_feature(
        #     clip_model, self.data_loader_test
        # )

    def get_feature(self, clip_model, data_loader):
        for batch_data in tqdm(data_loader):
            data = batch_data["img"].cuda()
            print(data)
            exit()


    def train(self):
        print("LinearProbe Train")
        exit()
