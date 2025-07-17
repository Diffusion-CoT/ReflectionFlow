import os
import json
from datetime import datetime
import time

import numpy as np
import torch
from diffusers import DiffusionPipeline
from tqdm.auto import tqdm
import copy
from PIL import Image
import time
from typing import Union, List, Optional
from verifiers.openai_verifier import OpenAIVerifier
from verifiers.nvila_verifier import load_model

from utils import get_noises, TORCH_DTYPE_MAP, get_latent_prep_fn, parse_cli_args

global verifier, yes_id, no_id
# Non-configurable constants
MAX_SEED = np.iinfo(np.int32).max  # To generate random seeds

def sample(
    noises: dict[int, torch.Tensor],
    original_prompt: str,
    updated_prompt: Union[str, List[str]],
    search_round: int,
    pipe: DiffusionPipeline,
    topk: int,
    root_dir: str,
    config: dict,
    midimg_path: str,
    tag: str,
) -> dict:
    """
    For a given prompt, generate images using all provided noises in batches,
    score them with the verifier, and select the top-K noise.
    The images and JSON artifacts are saved under `root_dir`.
    """
    global verifier, yes_id, no_id
    config_cp = copy.deepcopy(config)
    verifier_args = config["verifier_args"]
    verifier_name = verifier_args.get("name", "openai")

    refine_args = config["refine_args"]
    max_new_tokens = refine_args.get("max_new_tokens", None)
    choice_of_metric = refine_args.get("choice_of_metric", None)
    # currently only support openai refiner
    refiner = OpenAIVerifier(refine_prompt_relpath=refine_args["refine_prompt_relpath"], reflexion_prompt_relpath=refine_args["reflexion_prompt_relpath"], verifier_prompt_relpath=refine_args["verifier_prompt_relpath"])
    
    use_low_gpu_vram = config_cp.get("use_low_gpu_vram", False)
    batch_size_for_img_gen = config_cp.get("batch_size_for_img_gen", 1)

    images_for_prompt = []
    noises_used = []
    seeds_used = []
    prompts = updated_prompt

    # Convert the noises dictionary into a list of (seed, noise) tuples.
    noise_items = list(noises.items())

    # Process the noises in batches.
    full_imgnames = []
    times = []
    image_gen_times = []
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

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        batch_result = pipe(prompt=batched_prompts, latents=batched_latents, guidance_scale=config_cp["pipeline_args"]["guidance_scale"], num_inference_steps=config_cp["pipeline_args"]["num_inference_steps"], height=config_cp["pipeline_args"]["height"], width=config_cp["pipeline_args"]["width"])
        end.record()
        torch.cuda.synchronize()
        image_gen_times.extend([start.elapsed_time(end)])
        batch_images = batch_result.images
        if use_low_gpu_vram:
            pipe = pipe.to("cpu")

        # Iterate over the batch and save the images.
        start_time = time.time()
        for seed, noise, image, filename in zip(seeds_batch, noises_batch, batch_images, filenames_batch):
            images_for_prompt.append(image)
            noises_used.append(noise)
            seeds_used.append(seed)
            image.save(filename)
        end_time = time.time()
        times.append(end_time - start_time)

    print(f"Image serialization took: {sum(times):.2f} seconds.")

    # Prepare verifier inputs and perform inference.
    start_time = time.time()
    if verifier_name == "openai":
        verifier_inputs = verifier.prepare_inputs(images=images_for_prompt, prompts=[original_prompt]*len(images_for_prompt))
        outputs = verifier.score(
            inputs=verifier_inputs,
            tag=tag,
            max_new_tokens=max_new_tokens,  # Ignored when using Gemini for now.
        )
        def f(x):
            if isinstance(x[choice_of_metric], dict):
                return x[choice_of_metric]["score"]
            return x[choice_of_metric]
        sorted_list = sorted(outputs, key=lambda x: f(x), reverse=True)
    elif verifier_name == "nvila":
        outputs = []
        for imgname in full_imgnames:
            r1, scores1 = verifier.generate_content([Image.open(imgname), original_prompt])
            if r1 == "yes":
                outputs.append({"image_name": imgname, "label": "yes", "score": scores1[0][0, yes_id].detach().cpu().float().item()})
            else:
                outputs.append({"image_name": imgname, "label": "no", "score": scores1[0][0, no_id].detach().cpu().float().item()})
        def f(x):
            if x["label"] == "yes":
                return (0, -x["score"])
            else:
                return (1, x["score"])
        sorted_list = sorted(outputs, key=lambda x: f(x))
    end_time = time.time()
    print(f"Time taken for evaluation: {end_time - start_time} seconds")

    topk_scores = sorted_list[:topk]
    topk_idx = [outputs.index(x) for x in topk_scores]

    # Refine the prompt for the next round
    start_time = time.time()
    evaluations = [json.dumps(json_dict) for json_dict in outputs]
    if verifier_name == "openai":
        refined_prompt_inputs = refiner.prepare_refine_prompt_inputs(images=images_for_prompt, evaluations=evaluations, original_prompt=[original_prompt] * len(images_for_prompt), current_prompt=prompts)
    else:
        refined_prompt_inputs = refiner.prepare_refine_prompt_inputs(images=images_for_prompt, original_prompt=[original_prompt] * len(images_for_prompt), current_prompt=prompts)
    refined_prompt = refiner.refine_prompt(inputs=refined_prompt_inputs)
    assert len(refined_prompt) == len(prompts)
    prompts = refined_prompt

    with open(os.path.join(root_dir, f"best_img_meta.jsonl"), "a") as f:
        f.write(f"refined_prompt{search_round}: "+json.dumps(prompts) + "\n")
    end_time = time.time()
    print(f"Refinement with verifier and other IO took: {(end_time - start_time):.2f} seconds.")

    datapoint = {
        "original_prompt": original_prompt,
        "refined_prompt": prompts,
        "search_round": search_round,
        "num_noises": len(noises),
        "choice_of_metric": choice_of_metric,
        "image_gen_times": image_gen_times
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
    global verifier, yes_id, no_id
    args = parse_cli_args()
    os.environ["API_KEY"] = os.environ["OPENAI_API_KEY"]

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
    
    # Doesn't help that much currently as several things within the transformer are changing.
    if config["pipeline_args"].get("compile", False):
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        print("Compilation.")

    # Load the verifier model.
    verifier_args = config["verifier_args"]
    verifier_name = verifier_args.get("name", "openai")
    if verifier_name == "openai":
        verifier = OpenAIVerifier(refine_prompt_relpath=verifier_args["refine_prompt_relpath"], reflexion_prompt_relpath=verifier_args["reflexion_prompt_relpath"], verifier_prompt_relpath=verifier_args["verifier_prompt_relpath"])
    elif verifier_name == "nvila":
        verifier, yes_id, no_id = load_model(model_name=verifier_args["model_name"], cache_dir=verifier_args["cache_dir"])
    else:
        raise ValueError(f"Verifier {verifier_name} not supported")

    # warmup
    for _ in range(3):
        pipe("pok pok", num_inference_steps=10)

    # Main loop: For each search round and each prompt, generate images, verify, and save artifacts.
    with open(args.meta_path) as fp:
        metadatas = [json.loads(line) for line in fp]
    metadatas = metadatas[:1] # benchmarking

    # meta splits
    # if args.end_index == -1:
    #     metadatas = metadatas[args.start_index:]
    # else:
    #     metadatas = metadatas[args.start_index:args.end_index]
    
    timings = []
    for index, metadata in tqdm(enumerate(metadatas), desc="Sampling data"):
        # create output directory
        start_time = time.time()
        outpath = os.path.join(root_dir, f"{index + args.start_index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        # create middle img directory
        midimg_path = os.path.join(outpath, "samples")
        os.makedirs(midimg_path, exist_ok=True)

        # create metadata file
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)
        
        end_time = time.time()
        print(f"Folder creation and JSON opening took: {(end_time - start_time):.2f} seconds.")

        updated_prompt = [metadata['prompt']] * search_branch
        original_prompt = metadata['prompt']
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
            print(f"Number of noise samples: {len(noises)}")
            datapoint = sample(
                noises=noises,
                original_prompt=original_prompt,
                updated_prompt=updated_prompt,
                search_round=round,
                pipe=pipe,
                topk=search_branch,
                root_dir=outpath,
                config=config,
                midimg_path=midimg_path,
                tag=metadata['tag'],
            )
            timings.extend(datapoint["image_gen_times"])
            updated_prompt = datapoint['refined_prompt']

    print(f"Total image gen time: {sum(timings):.2f} seconds.")


if __name__ == "__main__":
    main()
