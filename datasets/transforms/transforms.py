from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
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
    transform_train = []

    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]

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
    transform_test = []

    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]

    transform_test += [Resize(max(cfg.INPUT.SIZE), interpolation=interp_mode)]

    transform_test += [CenterCrop(cfg.INPUT.SIZE)]

    transform_test += [ToTensor()]

    if "normalize" in transform_choices:
        transform_test += [
            Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ]

    transform_test = Compose(transform_test)
    print("Transform for Test: {}".format(transform_test))

    return transform_test
