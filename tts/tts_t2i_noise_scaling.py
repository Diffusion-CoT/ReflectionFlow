import os
import json

import numpy as np
import torch
from diffusers import DiffusionPipeline
from tqdm.auto import tqdm
import copy
from PIL import Image

from utils import get_noises, TORCH_DTYPE_MAP, get_latent_prep_fn, parse_cli_args

# Non-configurable constants
MAX_SEED = np.iinfo(np.int32).max  # To generate random seeds

def sample(
    noises: dict[int, torch.Tensor],
    prompts: list[str],
    search_round: int,
    pipe: DiffusionPipeline,
    config: dict,
    original_prompt: str,
    midimg_path: str,
) -> dict:
    """
    For a given prompt, generate images using all provided noises in batches,
    score them with the verifier, and select the top-K noise.
    The images and JSON artifacts are saved under `root_dir`.
    """
    config_cp = copy.deepcopy(config)
    
    use_low_gpu_vram = config_cp.get("use_low_gpu_vram", False)
    batch_size_for_img_gen = config_cp.get("batch_size_for_img_gen", 1)

    images_for_prompt = []
    noises_used = []
    seeds_used = []

    # Convert the noises dictionary into a list of (seed, noise) tuples.
    noise_items = list(noises.items())

    # Process the noises in batches.
    full_imgnames = []
    for i in range(0, len(noise_items), batch_size_for_img_gen):
        batch = noise_items[i : i + batch_size_for_img_gen]
        seeds_batch, noises_batch = zip(*batch)
        filenames_batch = [
            os.path.join(midimg_path, f"{search_round}_round@{seed}.png") for seed in seeds_batch
        ]
        full_imgnames.extend(filenames_batch)

        if use_low_gpu_vram:
            pipe = pipe.to("cuda:0")
        print(f"Generating images for batch with seeds: {[s for s in seeds_batch]}.")

        # Create a batched prompt list and stack the latents.
        batched_latents = torch.stack(noises_batch).squeeze(dim=1)
        batched_prompts = prompts[i : i + batch_size_for_img_gen]
        # breakpoint()
        batch_result = pipe(prompt=batched_prompts, latents=batched_latents, guidance_scale=config_cp["pipeline_args"]["guidance_scale"], num_inference_steps=config_cp["pipeline_args"]["num_inference_steps"], height=config_cp["pipeline_args"]["height"], width=config_cp["pipeline_args"]["width"])
        batch_images = batch_result.images
        if use_low_gpu_vram :
            pipe = pipe.to("cpu")

        # Iterate over the batch and save the images.
        for seed, noise, image, filename in zip(seeds_batch, noises_batch, batch_images, filenames_batch):
            images_for_prompt.append(image)
            noises_used.append(noise)
            seeds_used.append(seed)
            image.save(filename)
    
    datapoint = {
        "prompt": original_prompt,
        "search_round": search_round,
        "num_noises": len(noises),
    }
    return datapoint

@torch.no_grad()
def main():
    """
    Main function:
      - Parses CLI arguments.
      - Creates an output directory based on verifier and current datetime.
      - Loads prompts.
      - Loads the image-generation pipeline.
      - Loads the verifier model.
      - Runs several search rounds where for each prompt a pool of random noises is generated,
        candidate images are produced and verified, and the best noise is chosen.
    """
    args = parse_cli_args()
    os.environ["API_KEY"] = os.environ["OPENAI_API_KEY"] # args.openai_api_key

    # Build a config dictionary for parameters that need to be passed around.
    with open(args.pipeline_config_path, "r") as f:
        config = json.load(f)

    config.update(vars(args))

    search_rounds = config["search_args"]["search_rounds"]
    search_branch = config["search_args"]["search_branch"]

    # Create a root output directory: output/{verifier_to_use}/{current_datetime}
    pipeline_name = config["pipeline_args"].get("pretrained_model_name_or_path")
    cache_dir = config["pipeline_args"]["cache_dir"]
    root_dir = config["output_dir"]
    os.makedirs(root_dir, exist_ok=True)

    # Set up the image-generation pipeline (on the first GPU if available).
    torch_dtype = TORCH_DTYPE_MAP[config["pipeline_args"].get("torch_dtype")]
    pipe = DiffusionPipeline.from_pretrained(pipeline_name, torch_dtype=torch_dtype, cache_dir=cache_dir)
    if not config["use_low_gpu_vram"]:
        pipe = pipe.to("cuda:0")
    pipe.set_progress_bar_config(disable=True)

    # Main loop: For each search round and each prompt, generate images, verify, and save artifacts.
    with open(args.meta_path) as fp:
        metadatas = [json.loads(line) for line in fp]

    # meta splits
    if args.end_index == -1:
        metadatas = metadatas[args.start_index:]
    else:
        metadatas = metadatas[args.start_index:args.end_index]

    for index, metadata in tqdm(enumerate(metadatas), desc="Sampling prompts"):
        original_prompt = metadata['prompt']
        current_prompts = [original_prompt] * search_branch
        # create output directory
        outpath = os.path.join(root_dir, f"{index + args.start_index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        # create middle img directory
        midimg_path = os.path.join(outpath, "samples")
        os.makedirs(midimg_path, exist_ok=True)

        # create metadata file
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)
            
        for round in range(1, search_rounds + 1):
            print(f"\n=== Round: {round} ===")
            noises = get_noises(
                max_seed=MAX_SEED,
                num_samples=search_branch,
                height=config["pipeline_args"]["height"],
                width=config["pipeline_args"]["width"],
                dtype=torch_dtype,
                fn=get_latent_prep_fn(pipeline_name),
            )
            datapoint = sample(
                noises=noises,
                prompts=current_prompts,
                search_round=round,
                pipe=pipe,
                config=config,
                original_prompt=original_prompt,
                midimg_path=midimg_path,
            )


if __name__ == "__main__":
    main()
