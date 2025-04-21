import os
import json
from datetime import datetime

import numpy as np
import torch
from diffusers import DiffusionPipeline
from tqdm.auto import tqdm
import copy
from PIL import Image

from utils import prompt_to_filename, get_noises, TORCH_DTYPE_MAP, get_latent_prep_fn, parse_cli_args, MODEL_NAME_MAP

# Non-configurable constants
TOPK = 2  # Always selecting the top-1 noise for the next round
MAX_SEED = np.iinfo(np.int32).max  # To generate random seeds


def sample(
    noises: dict[int, torch.Tensor],
    prompts: list[str],
    search_round: int,
    pipe: DiffusionPipeline,
    verifier,
    topk: int,
    root_dir: str,
    config: dict,
    tag: str,
    original_prompt: str,
    midimg_path: str,
    sample_path_best: str=None,
    sample_path_best4: str=None,
    total_rounds: int=0
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

        if use_low_gpu_vram and verifier_to_use != "gemini":
            pipe = pipe.to("cuda:0")
        print(f"Generating images for batch with seeds: {[s for s in seeds_batch]}.")

        # Create a batched prompt list and stack the latents.
        batched_latents = torch.stack(noises_batch).squeeze(dim=1)
        batched_prompts = prompts[i : i + batch_size_for_img_gen]
        # breakpoint()
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

    # # Prepare verifier inputs and perform inference.
    # verifier_inputs = verifier.prepare_inputs(images=images_for_prompt, prompts=prompts)
    # print("Scoring with the verifier.")
    # outputs = verifier.score(
    #     inputs=verifier_inputs,
    #     tag=tag,
    #     max_new_tokens=max_new_tokens,  # Ignored when using Gemini for now.
    # )
    # def f(x):
    #     if isinstance(x[choice_of_metric], dict):
    #         return x[choice_of_metric]["score"]
    #     return x[choice_of_metric]
    
    # sorted_list = sorted(outputs, key=lambda x: f(x), reverse=True)
    # topk_scores = sorted_list[:topk]
    # topk_idx = [outputs.index(x) for x in topk_scores]

    # # save scores
    # for i, img_path in enumerate(full_imgnames):
    #     scores_dict[img_path] = outputs[i][choice_of_metric]['score']

    # # save the best img and best 4
    # if search_round == total_rounds:
    #     sorted_dict = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    #     for i in range(4):
    #         Image.open(sorted_dict[i][0]).save(os.path.join(sample_path_best4, f"{i:05}.png"))
    #     Image.open(sorted_dict[0][0]).save(os.path.join(sample_path_best, f"{search_round:05}.png"))
    
    datapoint = {
        "prompt": original_prompt,
        "search_round": search_round,
        "num_noises": len(noises),
        # "scores_dict": scores_dict,
        # "best_noise_seed": topk_scores[0]["seed"],
        # "best_score": topk_scores[0][choice_of_metric],
        # "choice_of_metric": choice_of_metric,
        # "best_img_path": best_img_path,
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

    for index, metadata in tqdm(enumerate(metadatas), desc="Sampling prompts"):
        original_prompt = metadata['prompt']
        current_prompts = [original_prompt] * 4
        # create output directory
        outpath = os.path.join(root_dir, f"{index + args.start_index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        # create middle img directory
        midimg_path = os.path.join(outpath, "midimg")
        os.makedirs(midimg_path, exist_ok=True)

        # # create sample img dir
        # sample_path_lastround = os.path.join(outpath, "samples_lastround")
        # os.makedirs(sample_path_lastround, exist_ok=True)
        # sample_path_best = os.path.join(outpath, "samples_best")
        # os.makedirs(sample_path_best, exist_ok=True)
        # sample_path_best4 = os.path.join(outpath, "samples_best4")
        # os.makedirs(sample_path_best4, exist_ok=True)

        # create metadata file
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)
            
        scores_dict = {}
        for round in range(1, search_rounds + 1):
            print(f"\n=== Round: {round} ===")
            noises = get_noises(
                max_seed=MAX_SEED,
                num_samples=TOPK,
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
                verifier=verifier,
                topk=TOPK,
                root_dir=root_dir,
                config=config,
                original_prompt=original_prompt,
                midimg_path=midimg_path,
                # sample_path_best=sample_path_best,
                # sample_path_best4=sample_path_best4,
                total_rounds=search_rounds,
                tag = metadata['tag'],
                # scores_dict=scores_dict
            )
            # scores_dict = datapoint['scores_dict']
        
        # # save best 4 of all 16
        # # Prepare verifier inputs and perform inference.
        # allimgs = []
        # for img in os.listdir(midimg_path):
        #     allimgs.append(os.path.join(midimg_path, img))
        # verifier_inputs = verifier.prepare_inputs(images=allimgs, prompts=[original_prompt]*len(allimgs))
        # print("Scoring with the verifier.")
        # outputs = verifier.score(
        #     inputs=verifier_inputs,
        #     max_new_tokens=1280,  # Ignored when using Gemini for now.
        # )
        # choice_of_metric = config["verifier_args"]["choice_of_metric"]
        # def f(x):
        #     if isinstance(x[choice_of_metric], dict):
        #         return x[choice_of_metric]["score"]
        #     return x[choice_of_metric]

        # sorted_list = sorted(outputs, key=lambda x: f(x), reverse=True)
        # topk_scores = sorted_list[:TOPK]
        # topk_idx = [outputs.index(x) for x in topk_scores]

        # best4imgs = [allimgs[i] for i in topk_idx]
        # for i in range(len(best4imgs)):
        #     best4imgs[i] = Image.open(best4imgs[i])
        #     best4imgs[i].save(os.path.join(sample_img, f"{i:05}.png"))


if __name__ == "__main__":
    main()
