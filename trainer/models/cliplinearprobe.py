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

        self.embedding_train, self.class_label_train = self.get_embedding(
            clip_model, self.data_loader_train
        )
        self.embedding_val, self.class_label_val = self.get_embedding(
            clip_model, self.data_loader_val
        )
        self.embedding_test, self.class_label_test = self.get_embedding(
            clip_model, self.data_loader_test
        )
        print(len(self.embedding_train))
        print(len(self.embedding_val))
        print(len(self.embedding_test))

    def get_embedding(self, clip_model, data_loader):
        embedding_list = []
        class_label_list = []

        for batch_data in tqdm(data_loader):
            data = batch_data["img"].cuda()
            embeddings = clip_model.encode_image(data).cpu()

            for embedding in embeddings:
                embedding_list.append(embedding.tolist())
            class_label_list.extend(batch_data["class_label"].tolist())

        return embedding_list, class_label_list

    def train(self):
        print("LinearProbe Train")
        exit()
