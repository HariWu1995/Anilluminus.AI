from copy import deepcopy

import numpy as np

from PIL import Image, ImageOps
from PIL.Image import Image as PILImage

from src.utils import find_divisible


def preprocess_image(image, mask=None, max_area: int = 500_000, downscale_step: float = 0.95):

    def find_divisible_by_8(*X, return_mode: str = 'nearest'):
        return [find_divisible(x, frac=8, return_mode=return_mode) for x in X]

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    elif not isinstance(image, PILImage):
        raise TypeError(f"{image.__class__} is not supported!")

    image = image.convert('RGB')
    W, H = image.size

    ## Auto-Scale
    W_new, H_new = find_divisible_by_8(W, H, return_mode='nearest')
    while (H_new * W_new) > max_area:
        W_new = int(W_new * downscale_step)
        H_new = int(H_new * downscale_step)
        W_new, H_new = find_divisible_by_8(W_new, H_new, return_mode='lower')

    image = image.resize((W_new, H_new))

    if mask is None:
        return image, (W, H)

    mask = deepcopy(mask)

    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)

    elif not isinstance(mask, PILImage):
        raise TypeError(f"{mask.__class__} is not supported!")

    mask = mask.convert('L').resize((W_new, H_new))
    return (image, mask), (W, H)

