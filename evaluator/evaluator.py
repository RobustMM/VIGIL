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
        pass

    def evaluate(self):
        pass
