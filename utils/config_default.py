from yacs.config import CfgNode as CN


def get_cfg_default():
    # ====================
    # Global CfgNode
    # ====================
    _C = CN()
    _C.OUTPUT_DIR = "./output/"
    _C.SEED = -1

    # ====================
    # Input CfgNode
    # ====================
    _C.INPUT = CN()
    _C.INPUT.SIZE = (224, 224)
    _C.INPUT.INTERPOLATION = "bilinear"
    _C.INPUT.TRANSFORMS = []
    _C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
    _C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
    # Transform Setting
    # Random Crop
    _C.INPUT.CROP_PADDING = 4
    # Random Resized Crop
    _C.INPUT.RRCROP_SCALE = (0.08, 1.0)
    # Cutout
    _C.INPUT.CUTOUT_N = 1
    _C.INPUT.CUTOUT_LEN = 16
    # Gaussian Noise
    _C.INPUT.GN_MEAN = 0.0
    _C.INPUT.GN_STD = 0.15
    # RandomAugment
    _C.INPUT.RANDAUGMENT_N = 2
    _C.INPUT.RANDAUGMENT_M = 10
    # ColorJitter (Brightness, Contrast, Saturation, Hue)
    _C.INPUT.COLORJITTER_B = 0.4
    _C.INPUT.COLORJITTER_C = 0.4
    _C.INPUT.COLORJITTER_S = 0.4
    _C.INPUT.COLORJITTER_H = 0.1
    # Random Gray Scale's Probability
    _C.INPUT.RGS_P = 0.2
    # Gaussian Blur
    _C.INPUT.GB_P = 0.5  # Probability of Applying Gaussian Blur
    _C.INPUT.GB_K = 21  # Kernel Size (Should be an Odd Number)

    # ====================
    # Dataset CfgNode
    # ====================
    _C.DATASET = CN()
    _C.DATASET.ROOT = ""
    _C.DATASET.NAME = ""
    _C.DATASET.SOURCE_DOMAINS = []
    _C.DATASET.TARGET_DOMAINS = []
    _C.DATASET.SUBSAMPLE_CLASSES = "all"

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
    _C.DATALOADER.TEST.BATCH_SIZE = 32

    # ====================
    # Model CfgNode
    # ====================
    _C.MODEL = CN()
    _C.MODEL.NAME = ""
    _C.MODEL.BACKBONE = ""

    # DomainMix
    _C.MODEL.DOMAINMIX = CN()
    _C.MODEL.DOMAINMIX.TYPE = "crossdomain"
    _C.MODEL.DOMAINMIX.ALPHA = 1.0
    _C.MODEL.DOMAINMIX.BETA = 1.0

    # CLIPZeroShot
    _C.MODEL.CLIPZeroShot = CN()
    _C.MODEL.CLIPZeroShot.BACKBONE = "ViT-B/32"

    # CLIPLinearProbe
    _C.MODEL.CLIPLinearProbe = CN()
    _C.MODEL.CLIPLinearProbe.BACKBONE = "ViT-B/32"

    # CoOp
    _C.MODEL.CoOp = CN()
    _C.MODEL.CoOp.BACKBONE = "ViT-B/32"
    _C.MODEL.CoOp.N_CTX = 16
    _C.MODEL.CoOp.CSC = False
    _C.MODEL.CoOp.PREC = "fp16"
    _C.MODEL.CoOp.CLASS_TOKEN_POSITION = "end"

    # CoCoOp
    _C.MODEL.CoCoOp = CN()
    _C.MODEL.CoCoOp.BACKBONE = "ViT-B/32"
    _C.MODEL.CoCoOp.N_CTX = 16
    _C.MODEL.CoCoOp.PREC = "fp16"

    # ====================
    # Optimizer CfgNode
    # ====================
    _C.OPTIM = CN()
    _C.OPTIM.NAME = "sgd"
    _C.OPTIM.LR = 0.0002
    _C.OPTIM.WEIGHT_DECAY = 5e-4
    _C.OPTIM.MOMENTUM = 0.9
    _C.OPTIM.SGD_DAMPENING = 0
    _C.OPTIM.SGD_NESTEROV = False
    _C.OPTIM.ADAM_BETA1 = 0.9
    _C.OPTIM.ADAM_BETA2 = 0.999
    _C.OPTIM.LR_SCHEDULER = "cosine"
    _C.OPTIM.STEP_SIZE = -1
    _C.OPTIM.GAMMA = 0.1  # Factor to reduce learning rate
    _C.OPTIM.MAX_EPOCH = 10
    _C.OPTIM.WARMUP_EPOCH = (
        -1  # Set WARMUP_EPOCH larger than 0 to activate warmup training
    )
    _C.OPTIM.WARMUP_TYPE = "linear"  # Either linear or constant
    _C.OPTIM.WARMUP_CONS_LR = 1e-5  # Constant learning rate when WARMUP_TYPE=constant
    _C.OPTIM.WARMUP_MIN_LR = 1e-5  # Minimum learning rate when WARMUP_TYPE=linear

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
    _C.TEST.SPLIT = "Test"
    _C.TEST.FINAL_Model = "last_step"

    return _C
