import numpy as np
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
        # Initialize start point of c for binary search
        search_list = [1e6, 1e4, 1e2, 1, 1e-2, 1e-4, 1e-6]
        acc_list = []
        for c in search_list:
            clf = LogisticRegression(
                solver="lbfgs", max_iter=1000, penalty="l2", C=c
            ).fit(self.embedding_train, self.class_label_train)
            pred = clf.predict(self.embedding_val)
            acc_val = sum(pred == self.class_label_val) / len(self.class_label_val)
            acc_list.append(acc_val)

        c_peak = search_list[np.argmax(acc_list)]
        c_left, c_right = 1e1 * c_peak, 1e1 * c_peak

        print(acc_list, flush=True)
        print("C Peak: {}".format(c_peak))

        exit()
