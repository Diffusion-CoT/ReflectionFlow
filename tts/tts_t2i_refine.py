import os
import json
from datetime import datetime

import numpy as np
import torch
from diffusers import DiffusionPipeline
from tqdm.auto import tqdm
import copy

from utils import prompt_to_filename, get_noises, TORCH_DTYPE_MAP, get_latent_prep_fn, parse_cli_args, MODEL_NAME_MAP

# Non-configurable constants
TOPK = 1  # Always selecting the top-1 noise for the next round
MAX_SEED = np.iinfo(np.int32).max  # To generate random seeds


def sample(
    noises: dict[int, torch.Tensor],
    original_prompt: str,
    updated_prompt: str,
    search_round: int,
    pipe: DiffusionPipeline,
    verifier,
    topk: int,
    root_dir: str,
    config: dict,
    round: int,
    sample_path: str,
    midimg_path: str,
    tag: str,
    total_rounds: int
) -> dict:
    """
    For a given prompt, generate images using all provided noises in batches,
    score them with the verifier, and select the top-K noise.
    The images and JSON artifacts are saved under `root_dir`.
    """
    config_cp = copy.deepcopy(config)
    
    verifier_args = config["verifier_args"]
    max_new_tokens = verifier_args.get("max_new_tokens", None)
    choice_of_metric = verifier_args.get("choice_of_metric", None)
    verifier_to_use = verifier_args.get("name", "gemini")
    
    use_low_gpu_vram = config_cp.get("use_low_gpu_vram", False)
    batch_size_for_img_gen = config_cp.get("batch_size_for_img_gen", 1)

    images_for_prompt = []
    noises_used = []
    seeds_used = []
    prompt_filename = prompt_to_filename(updated_prompt)
    num_samples = len(noises)
    prompts = [updated_prompt] * num_samples

    # Convert the noises dictionary into a list of (seed, noise) tuples.
    noise_items = list(noises.items())

    # Process the noises in batches.
    for i in range(0, len(noise_items), batch_size_for_img_gen):
        batch = noise_items[i : i + batch_size_for_img_gen]
        seeds_batch, noises_batch = zip(*batch)
        filenames_batch = [
            os.path.join(midimg_path, f"{search_round}_round@{seed}.png") for seed in seeds_batch
        ]

        if use_low_gpu_vram and verifier_to_use != "gemini":
            pipe = pipe.to("cuda:0")
        print(f"Generating images for batch with seeds: {[s for s in seeds_batch]}.")

        # Create a batched prompt list and stack the latents.
        batched_latents = torch.stack(noises_batch).squeeze(dim=1)
        batched_prompts = prompts[i : i + batch_size_for_img_gen]

        batch_result = pipe(prompt=batched_prompts, latents=batched_latents, guidance_scale=config_cp["pipeline_args"]["guidance_scale"], num_inference_steps=config_cp["pipeline_args"]["num_inference_steps"], height=config_cp["pipeline_args"]["height"], width=config_cp["pipeline_args"]["width"])
        batch_images = batch_result.images
        if use_low_gpu_vram and verifier_to_use != "gemini":
            pipe = pipe.to("cpu")

        # Iterate over the batch and save the images.
        for seed, noise, image, filename in zip(seeds_batch, noises_batch, batch_images, filenames_batch):
            images_for_prompt.append(image)
            noises_used.append(noise)
            seeds_used.append(seed)
            image.save(filename)

    # Prepare verifier inputs and perform inference.
    verifier_inputs = verifier.prepare_inputs(images=images_for_prompt, prompts=prompts)
    print("Scoring with the verifier.")
    outputs = verifier.score(
        inputs=verifier_inputs,
        tag=tag,
        max_new_tokens=max_new_tokens,  # Ignored when using Gemini for now.
    )
    # breakpoint()
    for o in outputs:
        assert choice_of_metric in o, o.keys()
    
    assert len(outputs) == len(images_for_prompt), (
        f"Expected len(outputs) to be same as len(images_for_prompt) but got {len(outputs)=} & {len(images_for_prompt)=}"
    )

    results = []
    for json_dict, seed_val, noise in zip(outputs, seeds_used, noises_used):
        # Attach the noise tensor so we can select top-K.
        merged = {**json_dict, "noise": noise, "seed": seed_val}
        results.append(merged)

    # Sort by the chosen metric descending and pick top-K.
    for x in results:
        assert choice_of_metric in x, (
            f"Expected all dicts in `results` to contain the `{choice_of_metric}` key; got {x.keys()}."
        )

    def f(x):
        if isinstance(x[choice_of_metric], dict):
            return x[choice_of_metric]["score"]
        return x[choice_of_metric]

    sorted_list = sorted(results, key=lambda x: f(x), reverse=True)
    topk_scores = sorted_list[:topk]
    topk_idx = [results.index(x) for x in topk_scores]

    # save the last round at sample path.
    if search_round == total_rounds:
        for i in range(len(images_for_prompt)):
            images_for_prompt[i].save(os.path.join(sample_path, f"{i:05}.png"))
    best_img_path = os.path.join(midimg_path, f"{search_round}_round@{topk_scores[0]['seed']}.png")
    
    # breakpoint()
    # Refine the prompt for the next round
    evaluations = [json.dumps(json_dict) for json_dict in outputs]
    refined_prompt_inputs = verifier.prepare_refine_prompt_inputs(images=images_for_prompt, evaluations=evaluations, original_prompt=[original_prompt] * len(images_for_prompt), current_prompt=prompts)
    refined_prompt = verifier.refine_prompt(inputs=refined_prompt_inputs)
    assert len(refined_prompt) == len(prompts)
    prompts = refined_prompt

    with open(os.path.join(root_dir, f"best_img_meta.jsonl"), "a") as f:
        f.write(f"refined_prompt{search_round}: "+json.dumps(prompts[topk_idx[0]]) + "\n")

    datapoint = {
        "original_prompt": original_prompt,
        "refined_prompt": prompts[topk_idx[0]],
        "search_round": search_round,
        "num_noises": len(noises),
        "best_noise_seed": topk_scores[0]["seed"],
        "best_score": topk_scores[0][choice_of_metric],
        "choice_of_metric": choice_of_metric,
        "best_img_path": best_img_path,
    }
    # # Save the best config JSON file alongside the images.
    # best_json_filename = best_img_path.replace(".png", ".json")
    # with open(best_json_filename, "w") as f:
    #     json.dump(datapoint, f, indent=4)
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
    os.environ["API_KEY"] = os.environ["OPENAI_API_KEY"]

    # Build a config dictionary for parameters that need to be passed around.
    with open(args.pipeline_config_path, "r") as f:
        config = json.load(f)
    
    config.update(vars(args))
    search_rounds = config["search_args"]["search_rounds"]

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
    verifier_name = verifier_args.get("name", "gemini")
    refine_prompt_relpath = verifier_args.get("refine_prompt_relpath", "refine_prompt.txt")
    reflexion_prompt_relpath = verifier_args.get("reflexion_prompt_relpath", "reflexion_prompt.txt")
    verifier_prompt_relpath = verifier_args.get("verifier_prompt_relpath", "verifier_prompt.txt")
    if verifier_name == "gemini":
        from verifiers.gemini_verifier import GeminiVerifier

        verifier = GeminiVerifier()
    elif verifier_name == "openai":
        from verifiers.openai_verifier import OpenAIVerifier

        verifier = OpenAIVerifier(refine_prompt_relpath=refine_prompt_relpath, reflexion_prompt_relpath=reflexion_prompt_relpath, verifier_prompt_relpath=verifier_prompt_relpath)
    else:
        from verifiers.qwen_verifier import QwenVerifier

        verifier = QwenVerifier(use_low_gpu_vram=verifier_args.get("use_low_gpu_vram", False))

    # Main loop: For each search round and each prompt, generate images, verify, and save artifacts.
    with open(args.meta_path) as fp:
        metadatas = [json.loads(line) for line in fp]

    # meta splits
    if args.end_index == -1:
        metadatas = metadatas[args.start_index:]
    else:
        metadatas = metadatas[args.start_index:args.end_index]

    for index, metadata in tqdm(enumerate(metadatas), desc="Sampling data"):
        # create output directory
        outpath = os.path.join(root_dir, f"{index + args.start_index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        # create sample directory
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        # create middle img directory
        midimg_path = os.path.join(outpath, "midimg")
        os.makedirs(midimg_path, exist_ok=True)

        # create metadata file
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        updated_prompt = metadata['prompt']
        original_prompt = metadata['prompt']
        for round in range(1, search_rounds + 1):
            print(f"\n=== Round: {round} ===")
            num_noises_to_sample = 4
            noises = get_noises(
                max_seed=MAX_SEED,
                num_samples=num_noises_to_sample,
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
                verifier=verifier,
                topk=TOPK,
                root_dir=outpath,
                config=config,
                round=round,
                sample_path=sample_path,
                midimg_path=midimg_path,
                tag=metadata['tag'],
                total_rounds=search_rounds,
            )
            updated_prompt = datapoint['refined_prompt']

if __name__ == "__main__":
    main()
