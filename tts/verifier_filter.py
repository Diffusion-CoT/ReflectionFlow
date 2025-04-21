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

from utils import prompt_to_filename, get_noises, TORCH_DTYPE_MAP, get_latent_prep_fn, parse_cli_args, MODEL_NAME_MAP

# Non-configurable constants
MAX_SEED = np.iinfo(np.int32).max  # To generate random seeds
MAX_RETRIES = 5
RETRY_DELAY = 2

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

    # Build a config dictionary for parameters that need to be passed around.
    with open(args.pipeline_config_path, "r") as f:
        config = json.load(f)
    
    config.update(vars(args))

    ### load nvila verifier for scoring
    verifier_args = config["verifier_args"]
    verifier, yes_id, no_id = load_model(model_name=verifier_args["model_name"], cache_dir=verifier_args["cache_dir"])

    metadatas = []
    for folder_name in sorted(os.listdir(args.imgpath)):
        folder_path = os.path.join(args.imgpath, folder_name)
        
        if os.path.isdir(folder_path):
            metadata_path = os.path.join(folder_path, 'metadata.jsonl')
            midimg_path = os.path.join(folder_path, 'midimg')

            with open(metadata_path, "r") as f:
                metadata = [json.loads(line) for line in f]
            folder_data = {
                'metadata': metadata,
                'images': []
            }

            round_images = {}
            for file in sorted(os.listdir(midimg_path)):
                if file.endswith('.png'):  
                    round_key = file.split('_round@')[0] 
                    if round_key not in round_images:
                        round_images[round_key] = []
                    round_images[round_key].append(file)

            # breakpoint()
            round_images = dict(sorted(round_images.items(), key=lambda x: int(x[0])))
            for round_key, files in round_images.items():
                for file in files: 
                    img_path = os.path.join(midimg_path, file)
                    folder_data['images'].append({'img_path': img_path})

            metadatas.append(folder_data)

    # meta splits
    if args.end_index == -1:
        metadatas = metadatas[args.start_index:]
    else:
        metadatas = metadatas[args.start_index:args.end_index]
    
    for index, metadata in tqdm(enumerate(metadatas), desc="Sampling data"):
        prompt = metadata['metadata'][0]['prompt']
        imgs = metadata['images']
        imgs = [tmp['img_path'] for tmp in imgs]
        cur_dir = os.path.dirname(imgs[0])
        cur_dir = os.path.dirname(cur_dir)
        nfe1_path = os.path.join(cur_dir, "nfe1")
        nfe2_path = os.path.join(cur_dir, "nfe2")
        nfe4_path = os.path.join(cur_dir, "nfe4")
        nfe8_path = os.path.join(cur_dir, "nfe8")
        nfe16_path = os.path.join(cur_dir, "nfe16")
        nfe32_path = os.path.join(cur_dir, "nfe32")
        os.makedirs(nfe1_path, exist_ok=True)
        os.makedirs(nfe2_path, exist_ok=True)
        os.makedirs(nfe4_path, exist_ok=True)
        os.makedirs(nfe8_path, exist_ok=True)
        os.makedirs(nfe16_path, exist_ok=True)
        os.makedirs(nfe32_path, exist_ok=True)

        start_time = time.time()

        outputs = []
        # nvila verifier
        for imgname in imgs:
            r1, scores1 = verifier.generate_content([Image.open(imgname), prompt])
            if r1 == "yes":
                outputs.append({"image_name": imgname, "label": "yes", "score": scores1[0][0, yes_id].detach().cpu().float().item()})
            else:
                outputs.append({"image_name": imgname, "label": "no", "score": scores1[0][0, no_id].detach().cpu().float().item()})

        end_time = time.time()
        print(f"Time taken for evaluation: {end_time - start_time} seconds")

        # nvila verfier filter rule
        def f(x):
            if x["label"] == "yes":
                return (0, -x["score"])
            else:
                return (1, x["score"])
            
        # do nfe1
        sorted_list_nfe1 = sorted(outputs[:1], key=lambda x: f(x))
        topk_scores_nfe1 = sorted_list_nfe1
        topk_idx_nfe1 = [outputs.index(x) for x in topk_scores_nfe1]
        selected_imgs_nfe1 = [imgs[i] for i in topk_idx_nfe1]
        img = Image.open(selected_imgs_nfe1[0])
        img.save(os.path.join(nfe1_path, f"{0:05}.png"))
        
        # do nfe2
        sorted_list_nfe2 = sorted(outputs[:2], key=lambda x: f(x))
        topk_scores_nfe2 = sorted_list_nfe2
        topk_idx_nfe2 = [outputs.index(x) for x in topk_scores_nfe2]
        selected_imgs_nfe2 = [imgs[i] for i in topk_idx_nfe2]
        img = Image.open(selected_imgs_nfe2[0])
        img.save(os.path.join(nfe2_path, f"{0:05}.png"))
        
        # do nfe4
        sorted_list_nfe4 = sorted(outputs[:4], key=lambda x: f(x))
        topk_scores_nfe4 = sorted_list_nfe4[:4]
        topk_idx_nfe4 = [outputs.index(x) for x in topk_scores_nfe4]
        selected_imgs_nfe4 = [imgs[i] for i in topk_idx_nfe4]
        img = Image.open(selected_imgs_nfe4[0])
        img.save(os.path.join(nfe4_path, f"{0:05}.png"))
        
        # do nfe8
        sorted_list_nfe8 = sorted(outputs[:8], key=lambda x: f(x))
        topk_scores_nfe8 = sorted_list_nfe8[:8]
        topk_idx_nfe8 = [outputs.index(x) for x in topk_scores_nfe8]
        selected_imgs_nfe8 = [imgs[i] for i in topk_idx_nfe8]
        img = Image.open(selected_imgs_nfe8[0])
        img.save(os.path.join(nfe8_path, f"{0:05}.png"))

        # do nfe16
        sorted_list_nfe16 = sorted(outputs[:16], key=lambda x: f(x))
        topk_scores_nfe16 = sorted_list_nfe16[:16]
        topk_idx_nfe16 = [outputs.index(x) for x in topk_scores_nfe16]
        selected_imgs_nfe16 = [imgs[i] for i in topk_idx_nfe16]
        img = Image.open(selected_imgs_nfe16[0])
        img.save(os.path.join(nfe16_path, f"{0:05}.png"))

        # do nfe32
        # breakpoint()
        sorted_list_nfe32 = sorted(outputs[:32], key=lambda x: f(x))
        topk_scores_nfe32 = sorted_list_nfe32[:32]
        topk_idx_nfe32 = [outputs.index(x) for x in topk_scores_nfe32]
        selected_imgs_nfe32 = [imgs[i] for i in topk_idx_nfe32]
        img = Image.open(selected_imgs_nfe32[0])
        img.save(os.path.join(nfe32_path, f"{0:05}.png"))


if __name__ == "__main__":
    main()
