from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)

INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
}


def build_transform(cfg, is_train=True):
    if is_train:
        return _build_transform_train(cfg, cfg.INPUT.TRANSFORMS)
    else:
        return _build_transform_test(cfg, cfg.INPUT.TRANSFORMS)


def _build_transform_train(cfg, transform_choices):
    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]

    transform_train = []

    if "random_resized_crop" in transform_choices:
        transform_train += [
            RandomResizedCrop(
                size=cfg.INPUT.SIZE,
                scale=cfg.INPUT.RRCROP_SCALE,
                interpolation=interp_mode,
            )
        ]

    if "random_flip" in transform_choices:
        transform_train += [RandomHorizontalFlip()]

    transform_train += [ToTensor()]

    if "normalize" in transform_choices:
        transform_train += [
            Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ]

    transform_train = Compose(transform_train)
    print("Transform for Train: {}".format(transform_train))

    return transform_train


def _build_transform_test(cfg, transform_choices):
    pass
