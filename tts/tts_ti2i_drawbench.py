import os
import json
from datetime import datetime

import numpy as np
import torch
from diffusers import DiffusionPipeline
from tqdm.auto import tqdm
import copy
from typing import Union, List, Optional
from PIL import Image
import sys
sys.path.append('/home/zhaol0c/uni_editing/ReflectionFlow')
from train_flux.src.flux.generate import generate
from train_flux.src.flux.condition import Condition
import time
# Import your models
from imscore.aesthetic.model import LAIONAestheticScorer
from imscore.hps.model import HPSv2
from imscore.preference.model import CLIPScore
from imscore.pickscore.model import PickScorer
from imscore.imreward.model import ImageReward
from torchvision import transforms as tfms

from utils import prompt_to_filename, get_noises, TORCH_DTYPE_MAP, get_latent_prep_fn, parse_cli_args, MODEL_NAME_MAP

# Non-configurable constants
TOPK = 1  # Always selecting the top-1 noise for the next round
MAX_SEED = np.iinfo(np.int32).max  # To generate random seeds
MAX_RETRIES = 5
RETRY_DELAY = 2

def extract_reflections(reflections: List[str]) -> List[str]:
    results = []
    for reflection in reflections:
        sections = reflection.split('\n\n')
        result = {}
        for section in sections:
            # split by title and content
            if ':  ' in section:
                title, content = section.split(':  ', 1)
                # remove the index number
                title = title.split('. ', 1)[1].strip()
                # split by \n-
                items = [item.strip() for item in content.split('\n-') if item.strip()]
                # store in dict
                result[title] = items
        results.append(result)
    return results

def concat_extract_reflections(reflections: List[str]) -> List[str]:
    results = []
    for reflection in reflections:
        sections = reflection.split('\n\n')
        # initialize result
        result = ""
        # iterate over each section
        for section in sections:
            # split by title and content
            if ':' in section:
                title, content = section.split(':', 1)
                # remove the index number
                title = title.split('.', 1)[1].strip()
                # split by \n-
                items = [item.strip() for item in content.split('\n-') if item.strip()]
                # if None is in items, then skip this section
                skipflag = False
                for item in items:
                    if "None" in item:
                        skipflag = True
                        break
                if skipflag:
                    continue
                # store in dict
                result += " ".join(items)
        results.append(result)
    return results

def sample(
    noises: dict[int, torch.Tensor],
    original_prompt: str,
    updated_prompt: str,
    reflections: Union[str, List[str]],
    search_round: int,
    pipe: DiffusionPipeline,
    verifier,
    topk: int,
    root_dir: str,
    config: dict,
    sample_path: str,
    imagetoupdate: Union[str, Image.Image],
    midimg_path: str,
    total_rounds: int,
    tag: Optional[str] = None,
    rewardmodel: Optional[any] = None
) -> dict:
    """
    For a given prompt, generate images using all provided noises in batches,
    score them with the verifier, and select the top-K noise.
    The images and JSON artifacts are saved under `root_dir`.
    """
    # breakpoint()
    flag_terminated = search_round == total_rounds
    config_cp = copy.deepcopy(config)
    
    verifier_args = config["verifier_args"]
    max_new_tokens = verifier_args.get("max_new_tokens", None)
    choice_of_metric = verifier_args.get("choice_of_metric", None)
    verifier_to_use = verifier_args.get("name", "gemini")
    
    use_low_gpu_vram = config_cp.get("use_low_gpu_vram", False)
    batch_size_for_img_gen = config_cp.get("batch_size_for_img_gen", 1)
    reflection_args = config_cp.get("reflection_args", None)

    images_for_prompt = []
    noises_used = []
    seeds_used = []
    num_samples = len(noises)
    
    original_prompts = [original_prompt] * num_samples

    if isinstance(imagetoupdate, str):
        imagetoupdate = Image.open(imagetoupdate)
    imagetoupdate = imagetoupdate.resize((config_cp["pipeline_args"]["condition_size"], config_cp["pipeline_args"]["condition_size"]))
    position_delta = np.array([0, -config_cp["pipeline_args"]["condition_size"] // 16])
    conditionimg = Condition(condition=imagetoupdate, condition_type="cot", position_delta=position_delta)

    # Convert the noises dictionary into a list of (seed, noise) tuples.
    noise_items = list(noises.items())

    # Process the noises in batches.
    # breakpoint()
    if reflection_args and reflection_args.get("run_reflection", False):
        prompts = [updated_prompt + " [Reflexion]: " + reflections[0]] * num_samples
    else:
        prompts = original_prompts
    start_time = time.time()
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
        batched_prompts = prompts[i : i + batch_size_for_img_gen]

        # use omini model to generate images
        batch_result = generate(
            pipe,
            prompt=batched_prompts,
            conditions=[conditionimg] ,
            height=config_cp["pipeline_args"]["height"],
            width=config_cp["pipeline_args"]["width"],
            model_config=config.get("model", None),
            default_lora=True,
        )
        batch_images = batch_result.images
        if use_low_gpu_vram and verifier_to_use != "gemini":
            pipe = pipe.to("cpu")

        # Iterate over the batch and save the images.
        for seed, noise, image, filename in zip(seeds_batch, noises_batch, batch_images, filenames_batch):
            images_for_prompt.append(image)
            noises_used.append(noise)
            seeds_used.append(seed)
            image.save(filename)
    end_time = time.time()
    print(f"Time taken for image generation: {end_time - start_time} seconds")

    # Prepare verifier inputs and perform inference.
    # breakpoint()
    start_time = time.time()
    verifier_inputs = verifier.prepare_inputs(images=images_for_prompt, prompts=original_prompts)
    print("Scoring with the verifier.")
    retries = 0
    if rewardmodel:
        transforms = [tfms.ToTensor()]
        transforms = tfms.Compose(transforms)
        tfs_img = []
        
        for img in images_for_prompt:
            tfs_img.append(transforms(img))
        pixels_tensor = torch.stack(tfs_img, dim=0).to("cuda")
        model_score = rewardmodel.score(pixels_tensor, original_prompts)
        list_scores = model_score.cpu().tolist()
        outputs = []
        for scores in list_scores:
            outputs.append({f"{choice_of_metric}": {"score":scores[0]}})
        # breakpoint()
    else:
        while retries < MAX_RETRIES:
            try:
                outputs = verifier.score(
                    inputs=verifier_inputs,
                    max_new_tokens=max_new_tokens,  # Ignored when using Gemini for now.
                )
                break
            except Exception as e:
                retries += 1
                print(f"Error scoring: {e}. Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
    end_time = time.time()
    print(f"Time taken for evaluation: {end_time - start_time} seconds")
    for o in outputs:
        assert choice_of_metric in o, f"{choice_of_metric} not in {o.keys()}"
    
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
    best_img_path = os.path.join(midimg_path, f"{search_round}_round@{topk_scores[0]['seed']}.png")

    # save best img evaluation results
    with open(os.path.join(root_dir, f"best_img_detailedscore.jsonl"), "a") as f:
        data = {
            "best_img_path": best_img_path,
            "evaluation": outputs,
            "filenames_batch": filenames_batch,
        }
        f.write(json.dumps(data) + "\n")

    # # save the last round imgs
    # if search_round == total_rounds:
    #     for i in range(len(images_for_prompt)):
    #         images_for_prompt[i].save(os.path.join(sample_path, f"{i:05}.png"))

    # save the best mean score img
    if search_round == 1:
        for i in range(len(images_for_prompt)):
            images_for_prompt[i].save(os.path.join(sample_path, f"{i:05}.png"))
    else:
        with open(os.path.join(root_dir, f"best_img_detailedscore.jsonl"), "r") as f:
            saved_scores = [json.loads(line) for line in f]
        mean_scores = []
        for group_score in saved_scores:
            mean_score = 0
            for score in group_score["evaluation"]:
                mean_score += score[choice_of_metric]["score"]
            mean_score /= len(group_score["evaluation"])
            mean_scores.append(mean_score)
        max_mean_score = max(mean_scores[:-1])
        if mean_scores[-1] > max_mean_score:
            for i in range(len(images_for_prompt)):
                images_for_prompt[i].save(os.path.join(sample_path, f"{i:05}.png"))
        else:
            # temperal forbid stop early
            flag_terminated = False
    
    # generate reflections first at each round
    # breakpoint()
    conditionimg_forreflections = [images_for_prompt[topk_idx[0]]]
    reflection_performed = False
    if reflection_args and reflection_args.get("run_reflection", False):
        start_time = time.time()
        evaluations = [json.dumps(outputs[topk_idx[0]])]
        reflection_inputs = verifier.prepare_reflexion_prompt_inputs(
            images=conditionimg_forreflections, 
            original_prompt=[original_prompt], 
            current_prompt=[updated_prompt], 
            reflections=reflections,
            evaluations=evaluations
        )
        print("Generating reflection.")
        retries = 0
        while retries < MAX_RETRIES:
            try:
                update_reflections = verifier.generate_reflections(
                    inputs=reflection_inputs,
                    max_new_tokens=max_new_tokens,  # Ignored when using Gemini for now.
                )
                break
            except Exception as e:
                retries += 1
                print(f"Error generating reflection: {e}. Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
        end_time = time.time()
        print(f"Time taken for reflection generation: {end_time - start_time} seconds")
        # update_reflections = extract_reflections(update_reflections)
        try:
            update_reflections = concat_extract_reflections(update_reflections)
        except Exception as e:
            print(f"Error concatenating reflections: {e}")
            update_reflections = update_reflections

        # we maintain updated_prompt and reflection separately, everytime we concat them together to form the flux input. and update seperately
        reflection_performed = True
    
    # breakpoint()
    # Refine the prompt for the next round
    prompt_refiner_args = config_cp.get("prompt_refiner_args", None)
    refinement_performed = False
    if prompt_refiner_args and prompt_refiner_args.get("run_refinement", False):
        start_time = time.time()
        # evaluations = [json.dumps(json_dict) for json_dict in outputs]
        evaluations = [json.dumps(outputs[topk_idx[0]])]
        refined_prompt_inputs = verifier.prepare_refine_prompt_inputs(images=[images_for_prompt[topk_idx[0]]], evaluations=evaluations, original_prompt=[original_prompt], current_prompt=[updated_prompt], reflections=update_reflections)
        refined_prompt = verifier.refine_prompt(inputs=refined_prompt_inputs)
        # assert len(refined_prompt) == len(prompts)
        prompts = refined_prompt
        end_time = time.time()
        print(f"Time taken for prompt refinement: {end_time - start_time} seconds")
        refinement_performed = True

    # best img reflections and refine prompt
    if reflection_performed:
        best_img_reflections = update_reflections * num_samples
    if refinement_performed:
        best_img_refine_prompt = prompts[0]
    # save mid meta results
    if reflection_performed or refinement_performed:
        with open(os.path.join(root_dir, f"best_img_meta.jsonl"), "a") as f:
            if reflection_performed:
                f.write(f"reflections{search_round}: "+json.dumps(best_img_reflections[0]) + "\n")
            if refinement_performed:
                f.write(f"refined_prompt{search_round}: "+json.dumps(best_img_refine_prompt) + "\n")

    datapoint = {
        "original_prompt": original_prompt,
        # "refined_prompt": best_img_refine_prompt,
        "search_round": search_round,
        "num_noises": len(noises),
        "best_noise_seed": topk_scores[0]["seed"],
        "best_score": topk_scores[0][choice_of_metric],
        "choice_of_metric": choice_of_metric,
        "best_img_path": best_img_path,
        "flag_terminated": flag_terminated,
        # "reflections": best_img_reflections,
    }
    if refinement_performed:
        datapoint["refined_prompt"] = best_img_refine_prompt
    if reflection_performed:
        datapoint["reflections"] = best_img_reflections
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
    os.environ["API_KEY"] = os.environ["OPENAI_API_KEY"] # args.openai_api_key

    # Build a config dictionary for parameters that need to be passed around.
    with open(args.pipeline_config_path, "r") as f:
        config = json.load(f)
    
    config.update(vars(args))

    search_rounds = config["search_args"]["search_rounds"]
    # num_prompts = args.num_prompts

    # Create a root output directory: output/{verifier_to_use}/{current_datetime}
    # current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
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

    if config["pipeline_args"].get("lora_path", None) is not None:
        pipe.load_lora_weights(config["pipeline_args"].get("lora_path"), adapter_name="reflection")
        print("LoRA loaded.")
        # pipe.fuse_lora(adapter_names=["reflection"])
        # pipe.unload_lora_weights()
    
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

    # # Main loop: For each search round and each prompt, generate images, verify, and save artifacts.
    # generate from geneval gt
    metadatas = []
    reflectionargs = config.get("reflection_args", None) 
    use_reflection = reflectionargs.get("run_reflection", False)
    refineargs = config.get("prompt_refiner_args", None) 
    use_refine = refineargs.get("run_refinement", False)
    for folder_name in sorted(os.listdir(args.imgpath)):
        folder_path = os.path.join(args.imgpath, folder_name)
        
        if os.path.isdir(folder_path):
            metadata_path = os.path.join(folder_path, 'metadata.jsonl')
            samples_path = os.path.join(folder_path, 'samples')

            with open(metadata_path, "r") as f:
                metadata = [json.loads(line) for line in f]
            folder_data = {
                'metadata': metadata,
                'images': []
            }

            if os.path.exists(samples_path):
                for file in sorted(os.listdir(samples_path)):
                    img_path = os.path.join(samples_path, file)
                    folder_data['images'].append({'img_path': img_path})
            metadatas.append(folder_data)


    # meta splits
    if args.end_index == -1:
        metadatas = metadatas[args.start_index:]
    else:
        metadatas = metadatas[args.start_index:args.end_index]

    imgreward_model = ImageReward.from_pretrained("RE-N-Y/ImageReward", cache_dir="/ibex/user/zhaol0c/uniediting/ttsexps/aesthetics_ckpt/imgreward")
    imgreward_model.eval()
    imgreward_model.to("cuda")

    for index, metadata in tqdm(enumerate(metadatas), desc="Sampling data"):
        metadatasave = metadata['metadata']
        images = metadata['images']
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
            json.dump(metadatasave[0], fp)

        for image in images:
            updated_prompt = metadatasave[0]['prompt']
            original_prompt = metadatasave[0]['prompt']
            num_noises_to_sample = 4 # this should be scaled like 2 ** search_round
            
            if use_reflection:
                reflections = [""] * num_noises_to_sample
            else:
                reflections = None
            
            imagetoupdate = image['img_path']
            for round in range(1, search_rounds + 1):
                print(f"\n=== Round: {round} ===")
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
                    reflections=reflections,
                    search_round=round,
                    pipe=pipe,
                    verifier=verifier,
                    topk=TOPK,
                    root_dir=outpath,
                    config=config,
                    sample_path=sample_path,
                    imagetoupdate=imagetoupdate,
                    midimg_path=midimg_path,
                    total_rounds=search_rounds,
                    rewardmodel=imgreward_model
                )
                if use_reflection or use_refine:
                    if use_reflection:
                        reflections = datapoint['reflections']
                    if use_refine:
                        updated_prompt = datapoint['refined_prompt']
                imagetoupdate = datapoint['best_img_path']
                if datapoint['flag_terminated']:
                    break
            break

if __name__ == "__main__":
    main()
