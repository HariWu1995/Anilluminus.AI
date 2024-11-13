
## Stable-Diffusion checkpoints for general purposes

sd15_ckpt_dirs = [
    './checkpoints/models', 
    'D:/stable-diffusion/sd-15/checkpoints',
]

sdxl_ckpt_dirs = [
    './checkpoints/models_xl', 
    'D:/stable-diffusion/sd-xl/checkpoints',
]

scheduler_dirs = [
    './checkpoints/scheduler',
]

ctrlnet_dirs = [
    './checkpoints/controlnets',
    'D:/stable-diffusion/sd-15/controlnet',
]

lora_dirs = [
    './checkpoints/loras',
    'D:/stable-diffusion/sd-15/Lora',
]

vae_dirs = [
    './checkpoints/vae',
    'D:/stable-diffusion/sd-15/VAE',
]


## Other checkpoints for general purposes

text_encodirs = dict(
    # ip_adapter = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    ip_adapter = "E:/MMM/clip-vit-h14-laion2B-s32B-b79K",
    # cs_composer = "E:/MMM/blip-image-captioning-large",
)

image_encodirs = dict(
    ip_adapter_15 = "E:/MMM/clip-vit-h14",
    ip_adapter_xl = "E:/MMM/clip-vit-bigg14",
    # iplus_comp_15 = "E:/MMM/clip-vit-h14",
    # iplus_comp_xl = "E:/MMM/clip-vit-h14",
)


## IP-Adapter for style transfer
## @ https://huggingface.co/h94/IP-Adapter

ip_adapteroot = "D:/stable-diffusion/IP-Adapter"
ip_adapters = dict(
    ip_adapter_15 = (ip_adapteroot, 'sd15', 'ip-adapter_sd15.safetensors'),
    ip_adapter_xl = (ip_adapteroot, 'sdxl', 'ip-adapter_sdxl.safetensors'),
    # iplus_comp_15 = (ip_adapteroot, 'sd15', 'ip-adapter-plus_sd15.safetensors'),
    # iplus_comp_xl = (ip_adapteroot, 'sdxl', 'ip-adapter-plus_sdxl.safetensors'),
    # cs_composer = "D:/stable-diffusion/Instant-Style/csgo_4_32.bin",
)


## AnimateDiff for animation
## @ https://huggingface.co/guoyww/animatediff

animatediroot = "E:/stable-diffusion/AnimateDiff"
animatediff = dict(
    ctrlnet = f"{animatediroot}/controlnet",
       lora = f"{animatediroot}/lora",
)


## Others
rembg_dir = 'D:/rembg'

