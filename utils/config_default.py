from yacs.config import CfgNode as CN


def get_cfg_default():
    _C = CN()
    _C.OUTPUT = "./output/"
    _C.SEED = -1
    _C.USE_CUDA = True

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
    