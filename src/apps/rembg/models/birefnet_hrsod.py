import os
import pooch

from . import BiRefNetGeneral


class BiRefNetHRSOD(BiRefNetGeneral):
    """
    This class represents a BiRefNet-HRSOD session, which is a subclass of BiRefNetGeneral.
    """

    @classmethod
    def download_models(cls, *args, **kwargs):
        """
        Downloads the BiRefNet-HRSOD model file from a specific URL and saves it.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The path to the downloaded model file.
        """
        fname = f"{cls.name(*args, **kwargs)}.onnx"
        pooch.retrieve(
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-HRSOD_DHU-epoch_115.onnx",
            None if cls.checksum_disabled(*args, **kwargs) else "md5:c017ade5de8a50ff0fd74d790d268dda",
            fname=fname,
            path=cls.ckpt_dir(*args, **kwargs),
            progressbar=True,
        )

        return os.path.join(cls.ckpt_dir(*args, **kwargs), fname)

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Returns the name of the BiRefNet-HRSOD session.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the session.
        """
        return "birefnet-hrsod"
