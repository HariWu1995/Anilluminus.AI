import os
import pooch

from typing import List

import numpy as np

from PIL import Image
from PIL.Image import Image as ImageClass

from .base import BaseModel


class U2netp(BaseModel):
    """
    This class represents a session for using the U2-Net† model.
    """
    def predict(self, img: ImageClass, *args, **kwargs) -> List[ImageClass]:
        """
        Predicts the mask for the given image using the U2netp model.

        Parameters:
            img (ImageClass): The input image.

        Returns:
            List[ImageClass]: The predicted mask.
        """
        ort_outs = self.inner_session.run(
            None,
            self.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (320, 320)),
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
        Downloads the U2netp model.

        Returns:
            str: The path to the downloaded model.
        """
        fname = f"{cls.name(*args, **kwargs)}.onnx"
        pooch.retrieve(
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
            None if cls.checksum_disabled(*args, **kwargs) else "md5:8e83ca70e441ab06c318d82300c84806",
            fname=fname,
            path=cls.ckpt_dir(*args, **kwargs),
            progressbar=True,
        )

        return os.path.join(cls.ckpt_dir(*args, **kwargs), fname)

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Returns the name of the U2netp model.

        Returns:
            str: The name of the model.
        """
        return "u2netp"
