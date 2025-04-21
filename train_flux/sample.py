import os
import json
import math
import random
import torch
import numpy as np
import argparse
from diffusers.pipelines import FluxPipeline, StableDiffusion3Pipeline
from src.flux.condition import Condition
from src.sd3.condition import Condition as ConditionSD3
from PIL import Image

from src.flux.generate import generate, seed_everything
from src.sd3.generate import generate as generate_sd3

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run FLUX pipeline with reflection prompts")
parser.add_argument("--model_name", type=str, default="flux", help="Model name")
parser.add_argument("--step", type=int, default=30, help="Number of inference steps")
parser.add_argument("--condition_size", type=int, default=1024, help="Size of condition image")
parser.add_argument("--target_size", type=int, default=1024, help="Size of target image")
parser.add_argument("--task_name", type=str, default="geneval", 
                    choices=["edit", "geneval", "flux_pro_short", "flux_pro_detailed", "flux_pro"],
                    help="Task name for selecting the appropriate dataset")
parser.add_argument("--lora_dir", type=str, 
                    default="/mnt/petrelfs/gaopeng/zl/ReflectionFlow/train_flux/runs/full_data_v2/20250227-040606/ckpt/5000/pytorch_lora_weights.safetensors",
                    help="Path to LoRA weights")
parser.add_argument("--output_dir", type=str, 
                    default="/mnt/petrelfs/gaopeng/zl/ReflectionFlow/train_flux/samples/full_data_1024_5k",
                    help="Base directory for output")
parser.add_argument("--root_dir", type=str, default="", help="Root directory for image paths")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale")
parser.add_argument("--image_guidance_scale", type=float, default=1.0, help="Guidance scale")

args = parser.parse_args()

# Set variables from parsed arguments
step = args.step
condition_size = args.condition_size
target_size = args.target_size
task_name = args.task_name
lora_dir = args.lora_dir
output_dir = args.output_dir
root_dir = args.root_dir

# Set json_dir based on task_name
if task_name == "edit":
    json_dir = "/mnt/petrelfs/gaopeng/zl/data/reflection/metadata/edit_reflection_cleaned_val.json"
    output_dir = os.path.join(output_dir, "edit")
elif task_name == "geneval":
    json_dir = "/mnt/petrelfs/zhuole/data/metadata_clean/geneval_pairs_val.json"
    output_dir = os.path.join(output_dir, "geneval")
elif task_name == "flux_pro_short":
    json_dir = "/mnt/petrelfs/gaopeng/zl/data/reflection/metadata/flux_pro_detailed_reflection_cleaned_val.json"
    output_dir = os.path.join(output_dir, "flux_pro_short")
elif task_name == "flux_pro_detailed":
    json_dir = "/mnt/petrelfs/gaopeng/zl/data/reflection/metadata/flux_pro_detailed_reflection_cleaned_val.json"
    output_dir = os.path.join(output_dir, "flux_pro_detailed")
elif task_name == "flux_pro":
    json_dir = "/mnt/petrelfs/gaopeng/zl/data/reflection/metadata/flux_pro_reflection_cleaned_val.json"
    output_dir = os.path.join(output_dir, "flux_pro")
else:
    raise ValueError(f"Invalid task name: {task_name}")
os.makedirs(output_dir, exist_ok=True)

if args.model_name == "flux":
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    )
else:
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.bfloat16, cache_dir = "/ibex/user/zhaol0c/uniediting/ReflectionFlow-main/sd3medium"
    )
pipe = pipe.to("cuda")
pipe.load_lora_weights(
    lora_dir,
    adapter_name="cot",
)

# Load the JSON list containing prompt and image details
print(f"Loading JSON from {json_dir}")
with open(json_dir, "r") as f:
    # items = [json.loads(line.strip()) for line in f]
    items = json.load(f)

seed_everything(args.seed)

for idx, item in enumerate(items):
    # Open the bad image and good image
    # bad_image_path = os.path.join(root_dir, item["bad_image"])
    # good_image_path = os.path.join(root_dir, item["good_image"])
    bad_image_path = item["bad_image"]
    good_image_path = item["good_image"]
    
    # Load images
    bad_image = Image.open(bad_image_path).convert("RGB")
    good_image = Image.open(good_image_path).convert("RGB")
    
    # Get dimensions
    bad_w, bad_h = bad_image.size
    good_w, good_h = good_image.size
    
    # Resize bad image to match good image dimensions
    bad_image = bad_image.resize((good_w, good_h), Image.BICUBIC)
    
    # Resize the shorter edge to target_size while maintaining aspect ratio
    ratio = target_size / min(good_w, good_h)
    new_w = math.ceil(good_w * ratio)
    new_h = math.ceil(good_h * ratio)
    
    # Resize both images to the same dimensions
    good_image = good_image.resize((new_w, new_h), Image.BICUBIC)
    bad_image = bad_image.resize((new_w, new_h), Image.BICUBIC)
    
    # Randomly crop both images to exactly target_size x target_size
    if new_w > target_size or new_h > target_size:
        left = random.randint(0, max(0, new_w - target_size))
        top = random.randint(0, max(0, new_h - target_size))
        
        # Apply the same crop to both images to maintain pixel correspondence
        good_image = good_image.crop((left, top, left + target_size, top + target_size))
        bad_image = bad_image.crop((left, top, left + target_size, top + target_size))
    
    # Finally, resize bad_image to condition_size
    image = bad_image.resize((condition_size, condition_size), Image.BICUBIC)
    
    # Create a condition for the pipeline
    if args.model_name == "flux":
        condition_cls = Condition
    else:
        condition_cls = ConditionSD3
    condition = condition_cls(
        condition_type="cot",
        condition=image,
        position_delta=np.array([0, -condition_size // 16])
    )
    
    # Build the prompt by combining base prompt and reflection prompt if available
    original_prompt = item["prompt"]
    prompt = item["prompt"]
    if "reflection_prompt" in item:
        prompt += " [Reflexion] " + item["reflection_prompt"]
    elif "instruction" in item:
        prompt += " [Reflexion] " + item["instruction"]
    elif "reflection" in item:
        prompt += " [Reflexion] " + item["reflection"]
    elif "edited_prompt_list" in item:
        prompt += " [Reflexion] " + item["edited_prompt_list"][-1]
    else:
        raise ValueError(f"No reflection found in item: {item}")
    
    # Generate the result image
    if args.model_name == "flux":
        generate_func = generate
    else:
        generate_func = generate_sd3
    # breakpoint()
    result_img = generate_func(
        pipe,
        prompt=original_prompt,
        prompt_2=prompt,
        conditions=[condition],
        num_inference_steps=step,
        height=target_size,
        width=target_size,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale
    ).images[0]

    # Concatenate bad image, good image, and generated image side by side
    concat_image = Image.new("RGB", (condition_size + target_size + target_size, target_size))
    concat_image.paste(image, (0, 0))
    concat_image.paste(good_image, (condition_size, 0))
    concat_image.paste(result_img, (condition_size + target_size, 0))
    
    # Save the concatenated image, using image_id if present
    output_name = item.get("image_id", f"result_{idx}")
    concat_image.save(os.path.join(output_dir, f"{output_name}.jpg"))