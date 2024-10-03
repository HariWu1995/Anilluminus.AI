import os
import yaml, json

from glob import glob
from typing import Union
from pathlib import Path


MODEL_EXTENSIONS = ['.safetensors','.ckpt','.pt','.pth']


def scan_checkpoint_dir(ckpt_dir: Union[str, Path], 
                        ckpt_exts: list = MODEL_EXTENSIONS, sub_model: bool = False):

    if not  isinstance(ckpt_dir, str):
        ckpt_dir = str(ckpt_dir)
    if  ckpt_dir.endswith('/'):
        ckpt_dir = ckpt_dir[:-1]

    ckpt_dict = dict()

    # Entire-folder
    for ckpt_name in os.listdir(ckpt_dir):
        if not sub_model:
            if not os.path.isdir(os.path.join(ckpt_dir, ckpt_name, 'unet')):
                continue
        elif (
            not any([
                    os.path.isfile(os.path.join(ckpt_dir, ckpt_name, ckpt)) 
                for ckpt in ['config.json','diffusion_pytorch_model.safetensors']
            ])
        ):
            continue
        ckpt_dict[ckpt_name] = os.path.join(ckpt_dir, ckpt_name)

    # Single-file
    ckpt_exts = [ext if ext.startswith('.') else f'.{ext}' for ext in ckpt_exts]
    for ext in ckpt_exts:
        ckpt_paths = glob(ckpt_dir + '/*' + ext)
        for path in ckpt_paths:
            ckpt_name = os.path.splitext(str(Path(path).name))[0]
            ckpt_dict[ckpt_name] = path

    return ckpt_dict


def find_divisible(n: int, frac: int, return_mode: str = 'nearest'):

    quotient = n // frac

    if return_mode == 'lower':
        return frac * quotient

    elif return_mode == 'upper':
        return frac * (quotient + 1)

    # Get the 2 nearest multiples of `frac``
    lower = frac * quotient
    upper = frac * (quotient + 1)
    
    # Find which multiple is closer
    if abs(n - lower) < abs(n - upper):
        return lower
    else:
        return upper


def prettify_dict(data: dict):
    print(yaml.dump(data, default_flow_style=False))


