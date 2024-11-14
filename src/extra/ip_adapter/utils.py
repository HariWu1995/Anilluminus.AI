from PIL import Image

import numpy as np

import torch
import torch.nn.functional as F


def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")


def get_generator(seed, device):
    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None
    return generator


def resize_width_height(width, height, min_short_side=512, max_long_side=1024):

    if width < height:
        if width < min_short_side:
            scale_factor = min_short_side / width
            new_width = min_short_side
            new_height = int(height * scale_factor)
        else:
            new_width, new_height = width, height

    else:
        if height < min_short_side:
            scale_factor = min_short_side / height
            new_width = int(width * scale_factor)
            new_height = min_short_side
        else:
            new_width, new_height = width, height

    if max(new_width, new_height) > max_long_side:
        scale_factor = max_long_side / max(new_width, new_height)
        new_width = int(new_width * scale_factor)
        new_height = int(new_height * scale_factor)

    return new_width, new_height


def resize_content(content_image):
    max_long_side = 1024
    min_short_side = 1024

    new_width, new_height = resize_width_height(content_image.size[0], 
                                                content_image.size[1],
                                                min_short_side=min_short_side, 
                                                max_long_side=max_long_side)
    height = new_height // 16 * 16
    width = new_width // 16 * 16
    content_image = content_image.resize((width, height))

    return width, height, content_image


attn_maps = {}

def hook_fn(name):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            attn_maps[name] = module.processor.attn_map
            del module.processor.attn_map
    return forward_hook


def register_cross_attention_hook(unet):
    for name, module in unet.named_modules():
        if name.split('.')[-1].startswith('attn2'):
            module.register_forward_hook(hook_fn(name))
    return unet


def upscale(attn_map, target_size):
    attn_map = torch.mean(attn_map, dim=0)
    attn_map = attn_map.permute(1,0)
    temp_size = None

    for i in range(0,5):
        scale = 2 ** i
        if ( target_size[0] // scale ) * ( target_size[1] // scale) == attn_map.shape[1]*64:
            temp_size = (target_size[0]//(scale*8), target_size[1]//(scale*8))
            break

    assert temp_size is not None, "temp_size cannot is None"

    attn_map = attn_map.view(attn_map.shape[0], *temp_size)
    attn_map = F.interpolate(attn_map.unsqueeze(0).to(dtype=torch.float32),
                             size=target_size,
                             mode='bilinear',
                             align_corners=False)[0]
    attn_map = torch.softmax(attn_map, dim=0)
    return attn_map


def get_net_attn_map(image_size, batch_size=2, instance_or_negative=False, detach=True):

    idx = 0 if instance_or_negative else 1
    net_attn_maps = []

    for name, attn_map in attn_maps.items():
        attn_map = attn_map.cpu() if detach else attn_map
        attn_map = torch.chunk(attn_map, batch_size)[idx].squeeze()
        attn_map = upscale(attn_map, image_size) 
        net_attn_maps.append(attn_map) 

    net_attn_maps = torch.mean(torch.stack(net_attn_maps, dim=0), dim=0)
    return net_attn_maps


def attnmaps2images(net_attn_maps):
    images = []
    for attn_map in net_attn_maps:
        attn_map = attn_map.cpu().numpy()
        normalized_attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
        normalized_attn_map = normalized_attn_map.astype(np.uint8)
        image = Image.fromarray(normalized_attn_map)
        images.append(image)
    return images
