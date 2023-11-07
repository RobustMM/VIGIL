from collections import OrderedDict

import numpy as np
from sklearn.metrics import f1_score
from tabulate import tabulate

from .build_evaluator import EVALUATOR_REGISTRY


@EVALUATOR_REGISTRY.register()
class Classification:
    def __init__(self, cfg, class_label_name_mapping=None):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []

    def process(self, model_output, ground_truth):
        pred = model_output.max(1)[1]
        matches = pred.eq(ground_truth).float()
        self._correct += int(matches.sum().item())
        self._total += ground_truth.shape[0]
        self._y_true.extend(ground_truth.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

    def evaluate(self):
        results = OrderedDict()
        accuracy = 100.0 * self._correct / self._total
        error_rate = 100.0 - accuracy
        macro_f1 = 100.0 * f1_score(
            self._y_true, self._y_pred, average="macro", labels=np.unique(self._y_true)
        )

        results["accuracy"] = accuracy
        results["error_rate"] = error_rate
        results["macro_f1"] = macro_f1

        evaluation_table = [
            ["Total #", f"{self._total:,}"],
            ["Correct #", f"{self._correct:,}"],
            ["Accuracy", f"{accuracy:.2f}%"],
            ["Error Rate", f"{error_rate:.2f}%"],
            ["Macro_F1", f"{macro_f1:.2f}%"],
        ]
        print(tabulate(evaluation_table))

        return results
