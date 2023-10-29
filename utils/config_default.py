from yacs.config import CfgNode as CN


def get_cfg_default():
    # ====================
    # Global CfgNode
    # ====================
    _C = CN()
    _C.OUTPUTS = "./outputs/"
    _C.SEED = -1

    # ====================
    # Input CfgNode
    # ====================
    _C.INPUT = CN()
    _C.INPUT.SIZE = (224, 224)
    _C.INPUT.INTERPOLATION = "bilinear"
    _C.INPUT.TRANSFORMS = ["normalize"]
    _C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
    _C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

    # ====================
    # Dataset CfgNode
    # ====================
    _C.DATASET = CN()
    _C.DATASET.ROOT = ""
    _C.DATASET.NAME = ""
    _C.DATASET.SOURCE_DOMAIN = []
    _C.DATASET.TARGET_DOMAIN = []

    # ====================
    # Dataloader CfgNode
    # ====================
    _C.DATALOADER = CN()
    _C.DATALOADER.NUM_WORKERS = 4
    # Setting for the train data loader
    _C.DATALOADER.TRAIN = CN()
    _C.DATALOADER.TRAIN.SAMPLER = "RandomSampler"
    _C.DATALOADER.TRAIN.BATCH_SIZE = 32
    # Setting for the test data loader
    _C.DATALOADER.TEST = CN()
    _C.DATALOADER.TEST.SAMPLER = "SequentialSampler"
    _C.DATALOADER.TEST.BATCH_SIZE = 64

    # ====================
    # Model CfgNode
    # ====================
    _C.MODEL = CN()
    _C.MODEL.NAME = "LinearProbe"
    _C.MODEL.BACKBONE = "RN50"

    # ====================
    # Optimizer CfgNode
    # ====================
    _C.OPTIM = CN()
    _C.OPTIM.NAME = "SGD"
    _C.OPTIM.LR = 5e-5
    _C.OPTIM.WEIGHT_DECAY = 5e-4
    _C.OPTIM.MOMENTUM = 0.9
    _C.OPTIM.BETA1 = 0.9
    _C.OPTIM.BETA2 = 0.999
    _C.OPTIM.LR_SCHEDULER = "cosine"
    _C.OPTIM.STEP_SIZE = -1
    _C.OPTIM.GAMMA = 0.1    # Factor to reduce learning rate
    _C.OPTIM.EPOCH = 10

    # ====================
    # Train CfgNode
    # ====================
    _C.TRAIN = CN()
    _C.TRAIN.PRINT_FREQ = 10

    # ====================
    # Test CfgNode
    # ====================
    _C.TEST = CN()
    _C.TEST.EVALUATOR = "Classification"
    _C.TEST.SPLIT = "test"
    _C.TEST.FINAL_Model = "last_step"

    return _C
