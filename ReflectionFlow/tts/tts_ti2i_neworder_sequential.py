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
from train_flux.src.sd3.generate import generate as generate_sd3
from train_flux.src.sd3.condition import Condition as ConditionSD3
import time
from transformers import AutoModel

from utils import prompt_to_filename, get_noises, TORCH_DTYPE_MAP, get_latent_prep_fn, parse_cli_args, MODEL_NAME_MAP

# nvila verifier
def load_model(model_name):
    global model, yes_id, no_id
    print("loading NVILA model")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="auto", cache_dir = "/ibex/user/zhaol0c/uniediting_continue/nvila")
    yes_id = model.tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = model.tokenizer.encode("no", add_special_tokens=False)[0]
    print("loading NVILA finished")

# Non-configurable constants
TOPK = 4  # Always selecting the top-1 noise for the next round
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
    updated_prompt: Union[str, List[str]],
    reflections: Union[str, List[str]],
    search_round: int,
    pipe: DiffusionPipeline,
    verifier,
    topk: int,
    root_dir: str,
    config: dict,
    sample_path_lastround: str,
    sample_path_best: str,
    sample_path_bestround: str,
    imagetoupdate: Union[str, Image.Image],
    midimg_path: str,
    total_rounds: int,
    chains: dict,
    tag: Optional[str] = None,
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

    model_name = config['pipeline_args']['pretrained_model_name_or_path']

    images_for_prompt = []
    noises_used = []
    seeds_used = []
    num_samples = len(noises)

    # Prepare verifier inputs and perform inference.
    # breakpoint()
    start_time = time.time()
    pil_imgs = [Image.open(tmp) for tmp in imagetoupdate]
    verifier_inputs = verifier.prepare_inputs(images=pil_imgs, prompts=[original_prompt]*len(imagetoupdate))
    print("Scoring with the verifier.")
    retries = 0
    while retries < MAX_RETRIES:
        try:
            outputs = verifier.score(
                inputs=verifier_inputs,
                tag=tag,
                max_new_tokens=max_new_tokens,  # Ignored when using Gemini for now.
            )
            break
        except Exception as e:
            retries += 1
            print(f"Error scoring: {e}. Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
    end_time = time.time()
    print(f"Time taken for evaluation: {end_time - start_time} seconds")

    def f(x):
        if isinstance(x[choice_of_metric], dict):
            return x[choice_of_metric]["score"]
        return x[choice_of_metric]

    # breakpoint()
    sorted_list = sorted(outputs, key=lambda x: f(x), reverse=True)
    topk_scores = sorted_list[:topk]
    topk_idx = [outputs.index(x) for x in topk_scores]
    # best_img_path = imagetoupdate[topk_idx[0]]
    selected_imgs = [imagetoupdate[i] for i in topk_idx]
    selected_outputs = [outputs[i] for i in topk_idx]
    if topk > len(selected_imgs):
        repeat_count = (topk - len(selected_imgs))
        selected_imgs = selected_imgs + selected_imgs[:repeat_count]
        selected_outputs = selected_outputs + selected_outputs[:repeat_count]

    # save best img evaluation results
    with open(os.path.join(root_dir, f"best_img_detailedscore.jsonl"), "a") as f:
        data = {
            # "best_img_path": best_img_path,
            "evaluation": selected_outputs,
            "filenames_batch": selected_imgs,
        }
        f.write(json.dumps(data) + "\n")
    #####################################################
    # generate reflections first at each round
    # breakpoint()
    conditionimg_forreflections = selected_imgs
    reflection_performed = False
    if reflection_args and reflection_args.get("run_reflection", False):
        start_time = time.time()
        # evaluations = [json.dumps(outputs[topk_idx[0]])]
        evaluations = [json.dumps(output_) for output_ in selected_outputs]
        reflection_inputs = verifier.prepare_reflexion_prompt_inputs(
            images=conditionimg_forreflections, 
            original_prompt=[original_prompt] * len(conditionimg_forreflections), 
            current_prompt=updated_prompt, 
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
    #####################################################
    # breakpoint()
    # Refine the prompt for the next round
    prompt_refiner_args = config_cp.get("prompt_refiner_args", None)
    refinement_performed = False
    if prompt_refiner_args and prompt_refiner_args.get("run_refinement", False):
        start_time = time.time()
        # evaluations = [json.dumps(json_dict) for json_dict in outputs]
        evaluations = [json.dumps(output_) for output_ in selected_outputs]
        refined_prompt_inputs = verifier.prepare_refine_prompt_inputs(images=selected_imgs, evaluations=evaluations, original_prompt=[original_prompt] * len(selected_imgs), current_prompt=updated_prompt, reflections=update_reflections)
        refined_prompt = verifier.refine_prompt(inputs=refined_prompt_inputs)
        # assert len(refined_prompt) == len(prompts)
        # prompts = refined_prompt
        end_time = time.time()
        print(f"Time taken for prompt refinement: {end_time - start_time} seconds")
        refinement_performed = True

    # best img reflections and refine prompt
    if reflection_performed:
        best_img_reflections = update_reflections 
    if refinement_performed:
        best_img_refine_prompt = refined_prompt
    # save mid meta results
    if reflection_performed or refinement_performed:
        with open(os.path.join(root_dir, f"best_img_meta.jsonl"), "a") as f:
            if reflection_performed:
                f.write(f"reflections{search_round}: "+json.dumps(best_img_reflections) + "\n")
            if refinement_performed:
                f.write(f"refined_prompt{search_round}: "+json.dumps(best_img_refine_prompt) + "\n")
            f.write(f"filenames_batch{search_round}: "+json.dumps(selected_imgs) + "\n")
    #####################################################
    # breakpoint()
    original_prompts = [original_prompt] * num_samples
    if model_name == "black-forest-labs/FLUX.1-dev":
        condition_cls = Condition
    else:
        condition_cls = ConditionSD3
    conditionimgs = []
    for i in range(len(selected_imgs)):
        tmp = Image.open(selected_imgs[i])
        tmp = tmp.resize((config_cp["pipeline_args"]["condition_size"], config_cp["pipeline_args"]["condition_size"]))
        position_delta = np.array([0, -config_cp["pipeline_args"]["condition_size"] // 16])
        conditionimgs.append(condition_cls(condition=tmp, condition_type="cot", position_delta=position_delta))

    # Convert the noises dictionary into a list of (seed, noise) tuples.
    noise_items = list(noises.items())

    # Process the noises in batches.
    # breakpoint()
    if reflection_args and reflection_args.get("run_reflection", False):
        if update_reflections:
            prompts = []
            for i in range(len(update_reflections)):
                prompts.append(refined_prompt[i] + " [Reflexion]: " + update_reflections[i])
        else: 
            prompts = refined_prompt
    else:
        prompts = original_prompts
    start_time = time.time()
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
        batched_prompts = prompts[i : i + batch_size_for_img_gen]
        conditionimgs_batch = conditionimgs[i : i + batch_size_for_img_gen]

        # use omini model to generate images
        if model_name == "black-forest-labs/FLUX.1-dev":
            generate_func = generate
        else:
            generate_func = generate_sd3
        # breakpoint()
        batch_result = generate_func(
            pipe,
            prompt=batched_prompts,
            conditions=conditionimgs_batch,
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

    # score again to decide whether save
    start_time = time.time()
    verifier_inputs = verifier.prepare_inputs(images=images_for_prompt, prompts=original_prompts)
    print("Scoring with the verifier.")
    retries = 0
    while retries < MAX_RETRIES:
        try:
            outputs = verifier.score(
                inputs=verifier_inputs,
                tag=tag,
                max_new_tokens=max_new_tokens,  # Ignored when using Gemini for now.
            )
            break
        except Exception as e:
            retries += 1
            print(f"Error scoring: {e}. Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
    end_time = time.time()
    print(f"Time taken for evaluation: {end_time - start_time} seconds")

    # init chain
    if search_round == 1:
        # breakpoint()
        # Update chains with the selected images and scores
        for i, img_path in enumerate(full_imgnames):
            if img_path not in chains:  # 如果链条中没有该图像，则初始化
                chains[img_path] = {"images": [], "scores": []}
            chains[img_path]["images"].append(img_path)  # 添加图像路径
            chains[img_path]["scores"].append(outputs[i][choice_of_metric]['score'])  # 添加得分
    # update chains
    else:
        # breakpoint()
        for i, (img_path, output) in enumerate(zip(full_imgnames, outputs)):
            parent_imgpath = selected_imgs[i]
            for img, score in chains.items():
                if parent_imgpath in chains[img]["images"]:
                    chains[img]["images"].append(img_path)  # 添加图像路径
                    chains[img]["scores"].append(outputs[i][choice_of_metric]['score'])  # 添加得分

    mean_score_curround = 0
    for output in outputs:
        mean_score_curround += output[choice_of_metric]['score']
    mean_score_curround /= len(outputs)


    # save the last round imgs
    if search_round == total_rounds:
        for i in range(len(images_for_prompt)):
            images_for_prompt[i].save(os.path.join(sample_path_lastround, f"{i:05}.png"))

    # save the best group img
    if search_round == 1:
        for i in range(len(images_for_prompt)):
            images_for_prompt[i].save(os.path.join(sample_path_bestround, f"{i:05}.png"))
    else:
        best_images = []
        for chain_key, chain in chains.items():
            best_idx = np.argmax(chain["scores"])  # Get the index of the best score
            best_images.append(chain["images"][best_idx])  # Save the corresponding image

        for i, img_path in enumerate(best_images):
            img = Image.open(img_path)
            img.save(os.path.join(sample_path_bestround, f"{i:05}.png"))

    # save the best 1 img
    if search_round ==total_rounds:
        all_scores_with_images = []
        for chain_key, chain in chains.items():
            for img_path, score in zip(chain["images"], chain["scores"]):
                all_scores_with_images.append((score, img_path))

        top_scores_with_images = sorted(all_scores_with_images, key=lambda x: x[0], reverse=True)[0]
        score, img_path = top_scores_with_images
        img = Image.open(img_path)
        img.save(os.path.join(sample_path_best, f"{i:05}.png"))

    datapoint = {
        "original_prompt": original_prompt,
        # "refined_prompt": best_img_refine_prompt,
        "search_round": search_round,
        "num_noises": len(noises),
        "choice_of_metric": choice_of_metric,
        "generated_img": full_imgnames,
        "flag_terminated": flag_terminated,
        "chains": chains,
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
                    folder_data['images'].append(img_path)
            metadatas.append(folder_data)

    # missindx = [25,51,77,103,129,155,181,207,233,259,285,311,337,363,389,415,441,467,493,519,545,552]
    # missindx = [545,552]

    # meta splits
    if args.end_index == -1:
        metadatas = metadatas[args.start_index:]
    else:
        metadatas = metadatas[args.start_index:args.end_index]

    for index, metadata in tqdm(enumerate(metadatas), desc="Sampling data"):
        # if index not in missindx:
        #     continue
        metadatasave = metadata['metadata']
        images = metadata['images']
        # create output directory
        outpath = os.path.join(root_dir, f"{index + args.start_index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        # create sample directory
        sample_path_lastround = os.path.join(outpath, "samples_lastround")
        os.makedirs(sample_path_lastround, exist_ok=True)
        sample_path_best = os.path.join(outpath, "samples_best")
        os.makedirs(sample_path_best, exist_ok=True)
        sample_path_bestround = os.path.join(outpath, "samples_bestround")
        os.makedirs(sample_path_bestround, exist_ok=True)

        # create middle img directory
        midimg_path = os.path.join(outpath, "midimg")
        os.makedirs(midimg_path, exist_ok=True)

        # create metadata file
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadatasave[0], fp)

        num_noises_to_sample = 2 # this should be scaled like 2 ** search_round
        updated_prompt = [metadatasave[0]['prompt']] * num_noises_to_sample
        original_prompt = metadatasave[0]['prompt']

        if use_reflection:
            reflections = [""] * num_noises_to_sample
        else:
            reflections = None

        imagetoupdate = images
        chains = {}
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
                topk=num_noises_to_sample,
                root_dir=outpath,
                config=config,
                sample_path_lastround=sample_path_lastround,
                sample_path_best=sample_path_best,
                sample_path_bestround=sample_path_bestround,
                imagetoupdate=imagetoupdate,
                midimg_path=midimg_path,
                tag=metadatasave[0]['tag'],
                total_rounds=search_rounds,
                chains=chains
            )
            if use_reflection or use_refine:
                if use_reflection:
                    reflections = datapoint['reflections']
                if use_refine:
                    updated_prompt = datapoint['refined_prompt']
            imagetoupdate = datapoint['generated_img']
            chains = datapoint['chains']
            if datapoint['flag_terminated']:
                break

if __name__ == "__main__":
    main()
