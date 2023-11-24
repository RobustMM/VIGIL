import os

import numpy as np
from clip import clip
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from trainer import MODEL_REGISTRY, Trainer


@MODEL_REGISTRY.register()
class CLIPLinearProbe(Trainer):
    def __init__(self, cfg):
        self.num_step = 8

        super().__init__(cfg=cfg)

    def build_model(self):
        print("Build CLIP LinearProbe")

        clip_model, _ = clip.load(
            self.cfg.MODEL.CLIPLinearProbe.BACKBONE,
            device=self.device,
            download_root=os.path.abspath(os.path.expanduser("data")),
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
            image = batch_data["img"].cuda()
            embeddings = clip_model.encode_image(image).cpu()

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
        print("C Peak: {}".format(c_peak))

        c_left, c_right = 1e-1 * c_peak, 1e1 * c_peak
        # Binary search for the best c
        for _ in range(self.num_step):
            c_left, c_right = self.binary_search(c_left, c_right)

        self.test()

    def binary_search(self, c_left, c_right):
        print("Binary Search")
        print("C Left: {}".format(c_left))
        print("C Right: {}".format(c_right))

        clf_left = LogisticRegression(
            solver="lbfgs", max_iter=1000, penalty="l2", C=c_left
        ).fit(self.embedding_train, self.class_label_train)
        pred_left = clf_left.predict(self.embedding_val)
        acc_left = sum(pred_left == self.class_label_val) / len(self.class_label_val)
        print("Val Accuracy (Left): {:.2f}".format(100 * acc_left))

        clf_right = LogisticRegression(
            solver="lbfgs", max_iter=1000, penalty="l2", C=c_right
        ).fit(self.embedding_train, self.class_label_train)
        pred_right = clf_right.predict(self.embedding_val)
        acc_right = sum(pred_right == self.class_label_val) / len(self.class_label_val)
        print("Val Accuracy (Right): {:.2f}".format(100 * acc_right))

        if acc_left < acc_right:
            self.model = clf_right
            c_left = 0.5 * (np.log10(c_right) + np.log10(c_left))
            c_right = np.log10(c_right)
        else:
            self.model = clf_left
            c_right = 0.5 * (np.log10(c_right) + np.log10(c_left))
            c_left = np.log10(c_left)

        self.test()

        return np.power(10, c_left), np.power(10, c_right)

    def test(self):
        pred_test = self.model.predict(self.embedding_test)
        acc_test = sum(pred_test == self.class_label_test) / len(self.class_label_test)
        print("Test Accuracy: {:.2f}".format(acc_test))
