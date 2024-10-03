---
library_name: diffusers
pipeline_tag: text-to-video
---

AnimateDiff is a method that allows you to create videos using pre-existing Stable Diffusion Text to Image models.

It achieves this by inserting motion module layers into a frozen text to image model and training it on video clips to extract a motion prior.
These motion modules are applied after the ResNet and Attention blocks in the Stable Diffusion UNet. Their purpose is to introduce coherent motion across image frames. To support these modules we introduce the concepts of a MotionAdapter and UNetMotionModel. These serve as a convenient way to use these motion modules with existing Stable Diffusion models.

<table>
    <tr>
        <td><center>
        masterpiece, bestquality, sunset.
        <br>
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-v3-euler-a.gif"
            alt="masterpiece, bestquality, sunset"
            style="width: 300px;" />
        </center></td>
    </tr>
</table>


The following example demonstrates how you can utilize the motion modules with an existing Stable Diffusion text to image model.

```python
import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils import export_to_gif

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-3")
# load SD 1.5 based finetuned model
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter)
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    beta_schedule="linear",
)
pipe.scheduler = scheduler

# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt=(
        "masterpiece, bestquality, highlydetailed, ultradetailed, sunset, "
        "orange sky, warm lighting, fishing boats, ocean waves seagulls, "
        "rippling water, wharf, silhouette, serene atmosphere, dusk, evening glow, "
        "golden hour, coastal landscape, seaside scenery"
    ),
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cpu").manual_seed(42),
)
frames = output.frames[0]
export_to_gif(frames, "animation.gif")
```
