---
library_name: diffusers
pipeline_tag: text-to-video
tags:
  - animatediff
---


AnimateDiff is a method that allows you to create videos using pre-existing Stable Diffusion Text to Image models.

It achieves this by inserting motion module layers into a frozen text to image model and training it on video clips to extract a motion prior.
These motion modules are applied after the ResNet and Attention blocks in the Stable Diffusion UNet. Their purpose is to introduce coherent motion across image frames. To support these modules we introduce the concepts of a MotionAdapter and UNetMotionModel. These serve as a convenient way to use these motion modules with existing Stable Diffusion models.

SparseControlNetModel is an implementation of ControlNet for [AnimateDiff](https://arxiv.org/abs/2307.04725).

ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.05543) by Lvmin Zhang, Anyi Rao, and Maneesh Agrawala.

The SparseCtrl version of ControlNet was introduced in [SparseCtrl: Adding Sparse Controls to Text-to-Video Diffusion Models](https://arxiv.org/abs/2311.16933) for achieving controlled generation in text-to-video diffusion models by Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai.

The following example demonstrates how you can utilize the motion modules and sparse controlnet with an existing Stable Diffusion text to image model.

<table align="center">
    <tr>
        <center>
          <b>closeup face photo of man in black clothes, night city street, bokeh, fireworks in background</b>
        </center>
    </tr>
    <tr>
        <td>
          <center>
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-firework.png" alt="closeup face photo of man in black clothes, night city street, bokeh, fireworks in background" />
          </center>
        </td>
        <td>
          <center>
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-sparsectrl-rgb-result.gif" alt="closeup face photo of man in black clothes, night city street, bokeh, fireworks in background" />
          </center>
        </td>
    </tr>
</table>

```python
import torch

from diffusers import AnimateDiffSparseControlNetPipeline
from diffusers.models import AutoencoderKL, MotionAdapter, SparseControlNetModel
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import export_to_gif, load_image


model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
motion_adapter_id = "guoyww/animatediff-motion-adapter-v1-5-3"
controlnet_id = "guoyww/animatediff-sparsectrl-rgb"
lora_adapter_id = "guoyww/animatediff-motion-lora-v1-5-3"
vae_id = "stabilityai/sd-vae-ft-mse"
device = "cuda"

motion_adapter = MotionAdapter.from_pretrained(motion_adapter_id, torch_dtype=torch.float16).to(device)
controlnet = SparseControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16).to(device)
vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16).to(device)
scheduler = DPMSolverMultistepScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    beta_schedule="linear",
    algorithm_type="dpmsolver++",
    use_karras_sigmas=True,
)
pipe = AnimateDiffSparseControlNetPipeline.from_pretrained(
    model_id,
    motion_adapter=motion_adapter,
    controlnet=controlnet,
    vae=vae,
    scheduler=scheduler,
    torch_dtype=torch.float16,
).to(device)
pipe.load_lora_weights(lora_adapter_id, adapter_name="motion_lora")

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-firework.png")

video = pipe(
    prompt="closeup face photo of man in black clothes, night city street, bokeh, fireworks in background",
    negative_prompt="low quality, worst quality",
    num_inference_steps=25,
    conditioning_frames=image,
    controlnet_frame_indices=[0],
    controlnet_conditioning_scale=1.0,
    generator=torch.Generator().manual_seed(42),
).frames[0]
export_to_gif(video, "output.gif")
```