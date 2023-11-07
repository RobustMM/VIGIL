from utils import Registry, check_availability

EVALUATOR_REGISTRY = Registry("EVALUATOR")


def build_evaluator(cfg, **kwargs):
    available_evaluators = EVALUATOR_REGISTRY.registered_names()
    check_availability(cfg.TEST.EVALUATOR, available_evaluators)
    print("Build Evaluator: {}".format(cfg.TEST.EVALUATOR))

    return EVALUATOR_REGISTRY.get(cfg.TEST.EVALUATOR)(cfg, **kwargs)
