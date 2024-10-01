import os
import pooch
from typing import List

import numpy as np

from PIL import Image
from PIL.Image import Image as ImageClass

from .base import BaseModel


class BiRefNetGeneral(BaseModel):
    """
    This class represents a BiRefNet-General session, which is a subclass of BaseModel.
    """
    def sigmoid(self, mat):
        return 1 / (1 + np.exp(-mat))

    def predict(self, img: ImageClass, *args, **kwargs) -> List[ImageClass]:
        """
        Predicts the output masks for the input image using the inner session.

        Parameters:
            img (ImageClass): The input image.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List[ImageClass]: The list of output masks.
        """
        ort_outs = self.inner_session.run(
            None,
            self.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (1024, 1024)),
        )

        pred = self.sigmoid(ort_outs[0][:, 0, :, :])

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
        Downloads the BiRefNet-General model file from a specific URL and saves it.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The path to the downloaded model file.
        """
        fname = f"{cls.name(*args, **kwargs)}.onnx"
        pooch.retrieve(
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx",
            None if cls.checksum_disabled(*args, **kwargs) else "md5:7a35a0141cbbc80de11d9c9a28f52697",
            fname=fname,
            path=cls.ckpt_dir(*args, **kwargs),
            progressbar=True,
        )

        return os.path.join(cls.ckpt_dir(*args, **kwargs), fname)

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Returns the name of the BiRefNet-General session.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the session.
        """
        return "birefnet-general"
