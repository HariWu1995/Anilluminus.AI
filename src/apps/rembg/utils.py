import io
from enum import Enum
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np
import onnxruntime as ort
from cv2 import (BORDER_DEFAULT, MORPH_ELLIPSE, MORPH_OPEN,
                 GaussianBlur, getStructuringElement, morphologyEx,)
from PIL import Image, ImageOps
from PIL.Image import Image as ImageClass

from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images

from scipy.ndimage import binary_erosion

from .session import new_session
from .models import sessions_class
from .models.base import BaseModel


ort.set_default_logger_severity(3)

kernel = getStructuringElement(MORPH_ELLIPSE, (3, 3))


class ReturnType(Enum):

    BYTES = 0
    PILLOW = 1
    NDARRAY = 2


def alpha_matting_cutout(
    img: ImageClass,
    mask: ImageClass,
    foreground_threshold: int,
    background_threshold: int,
    erode_structure_size: int,
) -> ImageClass:
    """
    Perform alpha matting on an image using a given mask and threshold values.

    This function takes a PIL image `img` and a PIL image `mask` as input, along with
    the `foreground_threshold` and `background_threshold` values used to determine
    foreground and background pixels. The `erode_structure_size` parameter specifies
    the size of the erosion structure to be applied to the mask.

    The function returns a PIL image representing the cutout of the foreground object
    from the original image.
    """
    if img.mode == "RGBA" or img.mode == "CMYK":
        img = img.convert("RGB")

    img_array = np.asarray(img)
    mask_array = np.asarray(mask)

    is_foreground = mask_array > foreground_threshold
    is_background = mask_array < background_threshold

    structure = None
    if erode_structure_size > 0:
        structure = np.ones((erode_structure_size, erode_structure_size), dtype=np.uint8)

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    trimap = np.full(mask_array.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    img_normalized = img_array / 255.0
    trimap_normalized = trimap / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)

    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    cutout = Image.fromarray(cutout)

    return cutout


def naive_cutout(img: ImageClass, mask: ImageClass) -> ImageClass:
    """
    Perform a simple cutout operation on an image using a mask.

    This function takes a PIL image `img` and a PIL image `mask` as input.
    It uses the mask to create a new image where the pixels from `img` are
    cut out based on the mask.

    The function returns a PIL image representing the cutout of the original
    image using the mask.
    """
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask)
    return cutout


def putalpha_cutout(img: ImageClass, mask: ImageClass) -> ImageClass:
    """
    Apply the specified mask to the image as an alpha cutout.

    Args:
        img (ImageClass): The image to be modified.
        mask (ImageClass): The mask to be applied.

    Returns:
        ImageClass: The modified image with the alpha cutout applied.
    """
    img.putalpha(mask)
    return img


def get_concat_v_multi(imgs: List[ImageClass]) -> ImageClass:
    """
    Concatenate multiple images vertically.

    Args:
        imgs (List[ImageClass]): The list of images to be concatenated.

    Returns:
        ImageClass: The concatenated image.
    """
    pivot = imgs.pop(0)
    for im in imgs:
        pivot = get_concat_v(pivot, im)
    return pivot


def get_concat_v(img1: ImageClass, img2: ImageClass) -> ImageClass:
    """
    Concatenate two images vertically.

    Args:
        img1 (ImageClass): The first image.
        img2 (ImageClass): The second image to be concatenated below the first image.

    Returns:
        ImageClass: The concatenated image.
    """
    dst = Image.new("RGBA", (img1.width, img1.height + img2.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (0, img1.height))
    return dst


def post_process(mask: np.ndarray) -> np.ndarray:
    """
    Post Process the mask for a smooth boundary by applying Morphological Operations
    Research based on paper: https://www.sciencedirect.com/science/article/pii/S2352914821000757
    args:
        mask: Binary Numpy Mask
    """
    mask = morphologyEx(mask, MORPH_OPEN, kernel)
    mask = GaussianBlur(mask, (5, 5), sigmaX=2, sigmaY=2, borderType=BORDER_DEFAULT)
    mask = np.where(mask < 127, 0, 255).astype(np.uint8)  # type: ignore
    return mask


def apply_background_color(img: ImageClass, color: Tuple[int, int, int, int]) -> ImageClass:
    """
    Apply the specified background color to the image.

    Args:
        img (ImageClass): The image to be modified.
        color (Tuple[int, int, int, int]): The RGBA color to be applied.

    Returns:
        ImageClass: The modified image with the background color applied.
    """
    r, g, b, a = color
    colored_image = Image.new("RGBA", img.size, (r, g, b, a))
    colored_image.paste(img, mask=img)

    return colored_image


def fix_image_orientation(img: ImageClass) -> ImageClass:
    """
    Fix the orientation of the image based on its EXIF data.

    Args:
        img (ImageClass): The image to be fixed.

    Returns:
        ImageClass: The fixed image.
    """
    return cast(ImageClass, ImageOps.exif_transpose(img))


def download_models() -> None:
    """
    Download models for image processing.
    """
    for session in sessions_class:
        session.download_models()


def remove_background(
    data: Union[bytes, ImageClass, np.ndarray],
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erosion_size: int = 10,
    session: Optional[Union[BaseModel, str]] = None,
    bgcolor: Optional[Tuple[int, int, int, int]] = None,
    only_mask: bool = False,
    post_process_mask: bool = False,
    force_return_bytes: bool = False,
    *args: Optional[Any],
    **kwargs: Optional[Any]
) -> Union[bytes, ImageClass, np.ndarray]:
    """
    Remove the background from an input image.

    This function takes in various parameters and returns a modified version of the input image with the background removed. The function can handle input data in the form of bytes, a PIL image, or a numpy array. The function first checks the type of the input data and converts it to a PIL image if necessary. It then fixes the orientation of the image and proceeds to perform background removal using the 'u2net' model. The result is a list of binary masks representing the foreground objects in the image. These masks are post-processed and combined to create a final cutout image. If a background color is provided, it is applied to the cutout image. The function returns the resulting cutout image in the format specified by the input 'return_type' parameter or as python bytes if force_return_bytes is true.

    Parameters:
        data (Union[bytes, ImageClass, np.ndarray]): The input image data.
        alpha_matting (bool, optional): Flag indicating whether to use alpha matting. Defaults to False.
        alpha_matting_foreground_threshold (int, optional): Foreground threshold for alpha matting. Defaults to 240.
        alpha_matting_background_threshold (int, optional): Background threshold for alpha matting. Defaults to 10.
        alpha_matting_erosion_size (int, optional): Erosion size for alpha matting. Defaults to 10.
        session (Optional[BaseModel], optional): A session object for the 'u2net' model. Defaults to None.
        only_mask (bool, optional): Flag indicating whether to return only the binary masks. Defaults to False.
        post_process_mask (bool, optional): Flag indicating whether to post-process the masks. Defaults to False.
        bgcolor (Optional[Tuple[int, int, int, int]], optional): Background color for the cutout image. Defaults to None.
        force_return_bytes (bool, optional): Flag indicating whether to return the cutout image as bytes. Defaults to False.
        *args (Optional[Any]): Additional positional arguments.
        **kwargs (Optional[Any]): Additional keyword arguments.

    Returns:
        Union[bytes, ImageClass, np.ndarray]: The cutout image with the background removed.
    """
    if isinstance(data, bytes) or force_return_bytes:
        return_type = ReturnType.BYTES
        img = cast(ImageClass, Image.open(io.BytesIO(cast(bytes, data))))

    elif isinstance(data, ImageClass):
        return_type = ReturnType.PILLOW
        img = cast(ImageClass, data)

    elif isinstance(data, np.ndarray):
        return_type = ReturnType.NDARRAY
        img = cast(ImageClass, Image.fromarray(data))

    else:
        raise ValueError(
            "Input type {} is not supported. Try using force_return_bytes=True to force python bytes output".format(type(data))
        )

    putalpha = kwargs.pop("putalpha", False)

    # Fix image orientation
    img = fix_image_orientation(img)

    if session is None:
        session = new_session("u2net", *args, **kwargs)
    elif isinstance(session, str):
        session = new_session(session, *args, **kwargs)

    masks = session.predict(img, *args, **kwargs)
    cutouts = []

    for mask in masks:
        if post_process_mask:
            mask = Image.fromarray(post_process(np.array(mask)))

        if only_mask:
            cutout = mask

        elif alpha_matting:
            try:
                cutout = alpha_matting_cutout(
                    img,
                    mask,
                    alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold,
                    alpha_matting_erosion_size,
                )

            except ValueError:
                if putalpha:
                    cutout = putalpha_cutout(img, mask)
                else:
                    cutout = naive_cutout(img, mask)

        else:
            if putalpha:
                cutout = putalpha_cutout(img, mask)
            else:
                cutout = naive_cutout(img, mask)

        cutouts.append(cutout)

    cutout = img
    if len(cutouts) > 0:
        cutout = get_concat_v_multi(cutouts)

    if bgcolor is not None and not only_mask:
        cutout = apply_background_color(cutout, bgcolor)

    # Hardcode
    return_type = ReturnType.PILLOW

    if return_type == ReturnType.PILLOW:
        return cutout

    if return_type == ReturnType.NDARRAY:
        return np.asarray(cutout)

    bio = io.BytesIO()
    cutout.save(bio, "PNG")
    bio.seek(0)

    return bio.read()

