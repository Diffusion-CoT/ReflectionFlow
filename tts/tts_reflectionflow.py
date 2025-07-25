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
sys.path.append('../')
from train_flux.flux.generate import generate
from train_flux.flux.condition import Condition
import time
from verifiers.openai_verifier import OpenAIVerifier
from verifiers.nvila_verifier import load_model

from utils import get_noises, TORCH_DTYPE_MAP, get_latent_prep_fn, parse_cli_args
from openai import OpenAI

global verifier, yes_id, no_id
client = OpenAI(api_key="0",base_url="http://0.0.0.0:8001/v1")

# our reflection model
def generate_messages(bad_image, prompt):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": bad_image}},
                {"type": "text", "text": f"Generate reflections to improve the input image according to the prompt. The prompt is: \"{prompt}\""},
            ]
        }
    ]
    return messages

# Non-configurable constants
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
    global verifier, yes_id, no_id
    # breakpoint()
    flag_terminated = search_round == total_rounds
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
    reflection_args = config_cp.get("reflection_args", None)

    images_for_prompt = []
    noises_used = []
    seeds_used = []
    num_samples = len(noises)

    # Prepare verifier inputs and perform inference.
    # breakpoint()
    start_time = time.time()
    outputs = []
    if verifier_name == "openai":
        pil_imgs = [Image.open(tmp) for tmp in imagetoupdate]
        verifier_inputs = verifier.prepare_inputs(images=pil_imgs, prompts=[original_prompt]*len(pil_imgs))
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
        for imgname in imagetoupdate:
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

    # breakpoint()
    topk_scores = sorted_list[:topk]
    topk_idx = [outputs.index(x) for x in topk_scores]
    selected_imgs = [imagetoupdate[i] for i in topk_idx]
    selected_outputs = [outputs[i] for i in topk_idx]
    if topk > len(selected_imgs):
        repeat_count = (topk - len(selected_imgs))
        selected_imgs = selected_imgs + selected_imgs[:repeat_count]
        selected_outputs = selected_outputs + selected_outputs[:repeat_count]

    # save best img evaluation results
    with open(os.path.join(root_dir, f"best_img_detailedscore.jsonl"), "a") as f:
        data = {
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
        evaluations = [json.dumps(output_) for output_ in selected_outputs]
        if reflection_args["name"] == "openai":
            reflection_inputs = refiner.prepare_reflexion_prompt_inputs(
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
                    update_reflections = refiner.generate_reflections(
                        inputs=reflection_inputs,
                        max_new_tokens=max_new_tokens,  # Ignored when using Gemini for now.
                    )
                    break
                except Exception as e:
                    retries += 1
                    print(f"Error generating reflection: {e}. Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
        else:
            reflection_inputs = []
            for idx in range(len(conditionimg_forreflections)):
                reflection_inputs.append(generate_messages(conditionimg_forreflections[idx], original_prompt))

            print("Generating reflection.")
            update_reflections = []
            for reflection_input in reflection_inputs:
                retries = 0
                while retries < MAX_RETRIES:
                    try:
                        res = client.chat.completions.create(messages=reflection_input, model="Qwen/Qwen2.5-VL-7B-Instruct")
                        update_reflections.append(res.choices[0].message.content)
                        break
                    except Exception as e: 
                        retries += 1
                        print(f"Error generating reflection: {e}. Retrying in {RETRY_DELAY} seconds...")
                        time.sleep(RETRY_DELAY)
        end_time = time.time()
        print(f"Time taken for reflection generation: {end_time - start_time} seconds")

        # we maintain updated_prompt and reflection separately, everytime we concat them together to form the flux input. and update seperately
        reflection_performed = True
    #####################################################
    # Refine the prompt for the next round
    prompt_refiner_args = config_cp.get("prompt_refiner_args", None)
    refinement_performed = False
    if prompt_refiner_args and prompt_refiner_args.get("run_refinement", False):
        start_time = time.time()
        evaluations = [json.dumps(output_) for output_ in selected_outputs]
        if verifier_name == "openai":
            refined_prompt_inputs = refiner.prepare_refine_prompt_inputs(images=selected_imgs, evaluations=evaluations, original_prompt=[original_prompt] * len(selected_imgs), current_prompt=updated_prompt, reflections=update_reflections)
        else:
            refined_prompt_inputs = refiner.prepare_refine_prompt_inputs(images=selected_imgs, original_prompt=[original_prompt] * len(selected_imgs), current_prompt=updated_prompt, reflections=update_reflections)
        refined_prompt = refiner.refine_prompt(inputs=refined_prompt_inputs)
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
    original_prompts = [original_prompt] * num_samples
    conditionimgs = []
    for i in range(len(selected_imgs)):
        tmp = Image.open(selected_imgs[i])
        tmp = tmp.resize((config_cp["pipeline_args"]["condition_size"], config_cp["pipeline_args"]["condition_size"]))
        position_delta = np.array([0, -config_cp["pipeline_args"]["condition_size"] // 16])
        conditionimgs.append(Condition(condition=tmp, condition_type="cot", position_delta=position_delta))

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

        if use_low_gpu_vram:
            pipe = pipe.to("cuda:0")
        print(f"Generating images for batch with seeds: {[s for s in seeds_batch]}.")

        # Create a batched prompt list and stack the latents.
        batched_prompts = prompts[i : i + batch_size_for_img_gen]
        conditionimgs_batch = conditionimgs[i : i + batch_size_for_img_gen]

        # use omini model to generate images
        batch_result = generate(
            pipe,
            prompt=batched_prompts,
            conditions=conditionimgs_batch,
            height=config_cp["pipeline_args"]["height"],
            width=config_cp["pipeline_args"]["width"],
            model_config=config.get("model", None),
            default_lora=True,
        )
        batch_images = batch_result.images
        if use_low_gpu_vram:
            pipe = pipe.to("cpu")

        # Iterate over the batch and save the images.
        for seed, noise, image, filename in zip(seeds_batch, noises_batch, batch_images, filenames_batch):
            images_for_prompt.append(image)
            noises_used.append(noise)
            seeds_used.append(seed)
            image.save(filename)
    end_time = time.time()
    print(f"Time taken for image generation: {end_time - start_time} seconds")

    ################## score again to decide whether save
    start_time = time.time()
    if verifier_name == "openai":
        verifier_inputs = verifier.prepare_inputs(images=images_for_prompt, prompts=[original_prompt]*len(images_for_prompt))
        outputs = verifier.score(
            inputs=verifier_inputs,
            tag=tag,
            max_new_tokens=max_new_tokens,  # Ignored when using Gemini for now.
        )
    elif verifier_name == "nvila":
        outputs = []
        for imgname in full_imgnames:
            r1, scores1 = verifier.generate_content([Image.open(imgname), original_prompt])
            if r1 == "yes":
                outputs.append({"image_name": imgname, "label": "yes", "score": scores1[0][0, yes_id].detach().cpu().float().item()})
            else:
                outputs.append({"image_name": imgname, "label": "no", "score": scores1[0][0, no_id].detach().cpu().float().item()})
    else:
        raise NotImplementedError(f"Verifier {verifier_name} not supported")
    end_time = time.time()
    print(f"Time taken for evaluation: {end_time - start_time} seconds")

    # init chain
    if search_round == 1:
        # Update chains with the selected images and scores
        if verifier_name == "openai":
            for i, img_path in enumerate(full_imgnames):
                if img_path not in chains: 
                    chains[img_path] = {"images": [], "scores": []}
                chains[img_path]["images"].append(img_path)  
                chains[img_path]["scores"].append(outputs[i][choice_of_metric]['score'])
        elif verifier_name == "nvila":
            for i, img_path in enumerate(full_imgnames):
                if img_path not in chains:  # init
                    chains[img_path] = {"images": [], "scores": [], "labels": []}
                chains[img_path]["images"].append(img_path)  
                chains[img_path]["labels"].append(outputs[i]["label"])
                chains[img_path]["scores"].append(outputs[i]["score"]) 
        else:
            raise NotImplementedError(f"Verifier {verifier_name} not supported")
    # update chains
    else:
        if verifier_name == "openai":
            for i, (img_path, output) in enumerate(zip(full_imgnames, outputs)):
                parent_imgpath = selected_imgs[i]
                for img, score in chains.items():
                    if parent_imgpath in chains[img]["images"]:
                        chains[img]["images"].append(img_path)
                        chains[img]["scores"].append(outputs[i][choice_of_metric]['score']) 
        elif verifier_name == "nvila":
            for i, (img_path, output) in enumerate(zip(full_imgnames, outputs)):
                parent_imgpath = selected_imgs[i]
                for img, score in chains.items():
                    if parent_imgpath in chains[img]["images"]:
                        chains[img]["images"].append(img_path) 
                        chains[img]["labels"].append(outputs[i]["label"])
                        chains[img]["scores"].append(outputs[i]["score"]) 
                        break
        else:
            raise NotImplementedError(f"Verifier {verifier_name} not supported")

    # save the last round imgs
    if search_round == total_rounds:
        for i in range(len(images_for_prompt)):
            images_for_prompt[i].save(os.path.join(sample_path_lastround, f"{i:05}.png"))

    # save the best img on each path
    if search_round == 1:
        for i in range(len(images_for_prompt)):
            images_for_prompt[i].save(os.path.join(sample_path_bestround, f"{i:05}.png"))
    else:
        # breakpoint()
        best_images = []
        for chain_key, chain in chains.items():
            if verifier_name == "openai":
                best_idx = np.argmax(chain["scores"]) 
            elif verifier_name == "nvila":
                best_idx = min(
                    range(len(chain["scores"])),
                    key=lambda idx: (0 if chain["labels"][idx] == "yes" else 1,  
                                -chain["scores"][idx] if chain["labels"][idx] == "yes" else chain["scores"][idx])
                )
            else:
                raise NotImplementedError(f"Verifier {verifier_name} not supported")
            best_images.append(chain["images"][best_idx])  # Save the corresponding image

        for i, img_path in enumerate(best_images):
            img = Image.open(img_path)
            img.save(os.path.join(sample_path_bestround, f"{i:05}.png"))

    # save the best 1 img
    if search_round ==total_rounds:
        all_scores_with_images = []
        if verifier_name == "openai":
            for chain_key, chain in chains.items():
                for img_path, score in zip(chain["images"], chain["scores"]):
                    all_scores_with_images.append((score, img_path))

            top_scores_with_images = sorted(all_scores_with_images, key=lambda x: x[0], reverse=True)[0]
        elif verifier_name == "nvila":
            for chain_key, chain in chains.items():
                for img_path, label, score in zip(chain["images"], chain['labels'], chain["scores"]):
                    all_scores_with_images.append((label, score, img_path))

            top_scores_with_images = sorted(all_scores_with_images,  key=lambda x: (0 if x[0] == "yes" else 1, -x[1] if x[0] == "yes" else x[1]))[0]
        else:
            raise NotImplementedError(f"Verifier {verifier_name} not supported")
        if verifier_name == "nvila":
            label, score, img_path = top_scores_with_images
        else:
            score, img_path = top_scores_with_images
        img = Image.open(img_path)
        img.save(os.path.join(sample_path_best, f"{i:05}.png"))

    datapoint = {
        "original_prompt": original_prompt,
        "search_round": search_round,
        "num_noises": len(noises),
        "choice_of_metric": choice_of_metric,
        "generated_img": full_imgnames,
        "flag_terminated": flag_terminated,
        "chains": chains,
    }
    if refinement_performed:
        datapoint["refined_prompt"] = best_img_refine_prompt
    if reflection_performed:
        datapoint["reflections"] = best_img_reflections
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
    os.environ["API_KEY"] = os.environ["OPENAI_API_KEY"] # args.openai_api_key

    # Build a config dictionary for parameters that need to be passed around.
    with open(args.pipeline_config_path, "r") as f:
        config = json.load(f)
    
    config.update(vars(args))

    search_rounds = config["search_args"]["search_rounds"]
    search_branch = config["search_args"]["search_branch"]

    pipeline_name = config["pipeline_args"].get("pretrained_model_name_or_path")
    cache_dir = config["pipeline_args"]["cache_dir"]
    root_dir = config["output_dir"]
    os.makedirs(root_dir, exist_ok=True)

    # Set up the image-generation pipeline (on the first GPU if available).
    torch_dtype = TORCH_DTYPE_MAP[config["pipeline_args"].get("torch_dtype")]
    pipe = DiffusionPipeline.from_pretrained(pipeline_name, torch_dtype=torch_dtype, cache_dir=cache_dir)
    if not config["use_low_gpu_vram"]:
        pipe = pipe.to("cuda")
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
    verifier_name = verifier_args.get("name", "openai")
    if verifier_name == "openai":
        verifier = OpenAIVerifier(refine_prompt_relpath=verifier_args["refine_prompt_relpath"], reflexion_prompt_relpath=verifier_args["reflexion_prompt_relpath"], verifier_prompt_relpath=verifier_args["verifier_prompt_relpath"])
    elif verifier_name == "nvila":
        verifier, yes_id, no_id = load_model(model_name=verifier_args["model_name"], cache_dir=verifier_args["cache_dir"])
    else:
        raise ValueError(f"Verifier {verifier_name} not supported")

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


    # meta splits
    if args.end_index == -1:
        metadatas = metadatas[args.start_index:]
    else:
        metadatas = metadatas[args.start_index:args.end_index]

    for index, metadata in tqdm(enumerate(metadatas), desc="Sampling data"):
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
        sample_path_bestround = os.path.join(outpath, "samples_path_bestround")
        os.makedirs(sample_path_bestround, exist_ok=True)

        # create middle img directory
        midimg_path = os.path.join(outpath, "midimg")
        os.makedirs(midimg_path, exist_ok=True)

        # create metadata file
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadatasave[0], fp)

        updated_prompt = [metadatasave[0]['prompt']] * search_branch
        original_prompt = metadatasave[0]['prompt']

        if use_reflection:
            reflections = [""] * search_branch
        else:
            reflections = None

        imagetoupdate = images
        chains = {}
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
                reflections=reflections,
                search_round=round,
                pipe=pipe,
                topk=search_branch,
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
