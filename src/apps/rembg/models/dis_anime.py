import os
import pooch

from typing import List

import numpy as np

from PIL import Image
from PIL.Image import Image as ImageClass

from .base import BaseModel


class Dis(BaseModel):
    """
    This class represents a session for object detection.
    """

    def predict(self, img: ImageClass, *args, **kwargs) -> List[ImageClass]:
        """
        Use a pre-trained model to predict the object in the given image.

        Parameters:
            img (ImageClass): The input image.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[ImageClass]: A list of predicted mask images.
        """
        ort_outs = self.inner_session.run(
            None,
            self.normalize(img, (0.485, 0.456, 0.406), (1.0, 1.0, 1.0), (1024, 1024)),
        )

        pred = ort_outs[0][:, 0, :, :]

        ma = np.max(pred)
        mi = np.min(pred)

        pred = (pred - mi) / (ma - mi)
        pred = np.squeeze(pred)

        mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
        mask = mask.resize(img.size, Image.Resampling.LANCZOS)

        return [mask]

    @classmethod
    def download_models(cls, *args, **kwargs):
        """
        Download the pre-trained models.

        Parameters:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: The path of the downloaded model file.
        """
        fname = f"{cls.name(*args, **kwargs)}.onnx"
        pooch.retrieve(
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx",
            None if cls.checksum_disabled(*args, **kwargs) else "md5:6f184e756bb3bd901c8849220a83e38e",
            fname=fname,
            path=cls.ckpt_dir(*args, **kwargs),
            progressbar=True,
        )

        return os.path.join(cls.ckpt_dir(*args, **kwargs), fname)

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Get the name of the pre-trained model.

        Parameters:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: The name of the pre-trained model.
        """
        return "isnet-anime"
