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
sys.path.append('/home/zhaol0c/uni_editing/ReflectionFlow/reward_modeling')
from train_flux.src.flux.generate import generate
from train_flux.src.flux.condition import Condition
import time
from transformers import AutoModel
# from reward_modeling.test_reward import VideoVLMRewardInference

from utils import prompt_to_filename, get_noises, TORCH_DTYPE_MAP, get_latent_prep_fn, parse_cli_args, MODEL_NAME_MAP

# Non-configurable constants
MAX_SEED = np.iinfo(np.int32).max  # To generate random seeds
MAX_RETRIES = 5
RETRY_DELAY = 2

# # our verifier
# score_verfier = VideoVLMRewardInference("/ibex/user/zhaol0c/uniediting_continue/our_reward/models--diffusion-cot--reward-models/snapshots/5feb9ad5db2048b645178804eeb326c93039daa6", load_from_pretrained_step=10080, device="cuda", dtype=torch.bfloat16)

# nvila verifier
def load_model(model_name):
    global model, yes_id, no_id
    print("loading NVILA model")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="auto", cache_dir = "/ibex/user/zhaol0c/uniediting_continue/nvila")
    yes_id = model.tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = model.tokenizer.encode("no", add_special_tokens=False)[0]
    print("loading NVILA finished")

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
    load_model(model_name="Efficient-Large-Model/NVILA-Lite-2B-Verifier")

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
            midimg_path = os.path.join(folder_path, 'samples')

            with open(metadata_path, "r") as f:
                metadata = [json.loads(line) for line in f]
            folder_data = {
                'metadata': metadata,
                'images': []
            }

            round_images = []
            for file in sorted(os.listdir(midimg_path)):
                round_images.append(file)
                    
            # 按文件名排序（按照数字顺序）
            # breakpoint()
            round_images = sorted(round_images, key=lambda x: int(x.split('.')[0]))

            # 添加到 folder_data
            for file in round_images:
                img_path = os.path.join(midimg_path, file)
                folder_data['images'].append({'img_path': img_path})

            # if os.path.exists(midimg_path):
            #     for file in sorted(os.listdir(midimg_path)):
            #         img_path = os.path.join(midimg_path, file)
            #         folder_data['images'].append({'img_path': img_path})
            metadatas.append(folder_data)

    # meta splits
    if args.end_index == -1:
        metadatas = metadatas[args.start_index:]
    else:
        metadatas = metadatas[args.start_index:args.end_index]
    
    for index, metadata in tqdm(enumerate(metadatas), desc="Sampling data"):
        # breakpoint()
        prompt = metadata['metadata'][0]['prompt']
        tag = metadata['metadata'][0]['tag']
        imgs = metadata['images']
        imgs = [tmp['img_path'] for tmp in imgs]
        cur_dir = os.path.dirname(imgs[0])
        cur_dir = os.path.dirname(cur_dir)
        # sample_path = os.path.join(cur_dir, "selected_best4")
        # nfe1_path = os.path.join(cur_dir, "nfe1")
        # nfe2_path = os.path.join(cur_dir, "nfe2")
        # nfe4_path = os.path.join(cur_dir, "nfe4")
        # nfe8_path = os.path.join(cur_dir, "nfe8")
        # nfe16_path = os.path.join(cur_dir, "nfe16")
        # nfe32_path = os.path.join(cur_dir, "nfe32")
        # # os.makedirs(sample_path, exist_ok=True)
        # # os.makedirs(nfe1_path, exist_ok=True)
        # os.makedirs(nfe2_path, exist_ok=True)
        # os.makedirs(nfe4_path, exist_ok=True)
        # os.makedirs(nfe8_path, exist_ok=True)
        # os.makedirs(nfe16_path, exist_ok=True)
        # os.makedirs(nfe32_path, exist_ok=True)

        nfe20_select4_path = os.path.join(cur_dir, "nfe20_select4")
        nfe20_path = os.path.join(cur_dir, "nfe20")
        os.makedirs(nfe20_select4_path, exist_ok=True)
        os.makedirs(nfe20_path, exist_ok=True)

        start_time = time.time()

        outputs = []
        # # our verifier
        # for imgname in imgs:
        #     scores = score_verfier.reward([imgname], [prompt], use_norm=True)
        #     outputs.append({"image_name": imgname, "score": scores[0]["VQ"]})

        # nvila verifier
        for imgname in imgs:
            r1, scores1 = model.generate_content([Image.open(imgname), prompt])
            if r1 == "yes":
                outputs.append({"image_name": imgname, "label": "yes", "score": scores1[0][0, yes_id].detach().cpu().float().item()})
            else:
                outputs.append({"image_name": imgname, "label": "no", "score": scores1[0][0, no_id].detach().cpu().float().item()})

        end_time = time.time()
        print(f"Time taken for evaluation: {end_time - start_time} seconds")
        # breakpoint()
        # # our verifier filter rule
        # def f(x):
        #     return (0, -x["score"])
        
        # nvila verfier filter rule
        def f(x):
            if x["label"] == "yes":
                return (0, -x["score"])
            else:
                return (1, x["score"])
        
        # # do nfe2
        # sorted_list_nfe2 = sorted(outputs[:2], key=lambda x: f(x))
        # topk_scores_nfe2 = sorted_list_nfe2
        # topk_idx_nfe2 = [outputs.index(x) for x in topk_scores_nfe2]
        # selected_imgs_nfe2 = [imgs[i] for i in topk_idx_nfe2]
        # img = Image.open(selected_imgs_nfe2[0])
        # img.save(os.path.join(nfe2_path, f"{0:05}.png"))
        
        # # do nfe4
        # sorted_list_nfe4 = sorted(outputs[:4], key=lambda x: f(x))
        # topk_scores_nfe4 = sorted_list_nfe4[:4]
        # topk_idx_nfe4 = [outputs.index(x) for x in topk_scores_nfe4]
        # selected_imgs_nfe4 = [imgs[i] for i in topk_idx_nfe4]
        # img = Image.open(selected_imgs_nfe4[0])
        # img.save(os.path.join(nfe4_path, f"{0:05}.png"))
        
        # # do nfe8
        # sorted_list_nfe8 = sorted(outputs[:8], key=lambda x: f(x))
        # topk_scores_nfe8 = sorted_list_nfe8[:8]
        # topk_idx_nfe8 = [outputs.index(x) for x in topk_scores_nfe8]
        # selected_imgs_nfe8 = [imgs[i] for i in topk_idx_nfe8]
        # img = Image.open(selected_imgs_nfe8[0])
        # img.save(os.path.join(nfe8_path, f"{0:05}.png"))

        # # do nfe16
        # sorted_list_nfe16 = sorted(outputs[:16], key=lambda x: f(x))
        # topk_scores_nfe16 = sorted_list_nfe16[:16]
        # topk_idx_nfe16 = [outputs.index(x) for x in topk_scores_nfe16]
        # selected_imgs_nfe16 = [imgs[i] for i in topk_idx_nfe16]
        # img = Image.open(selected_imgs_nfe16[0])
        # img.save(os.path.join(nfe16_path, f"{0:05}.png"))

        # # do nfe32
        # # breakpoint()
        # sorted_list_nfe32 = sorted(outputs[:32], key=lambda x: f(x))
        # topk_scores_nfe32 = sorted_list_nfe32[:32]
        # topk_idx_nfe32 = [outputs.index(x) for x in topk_scores_nfe32]
        # selected_imgs_nfe32 = [imgs[i] for i in topk_idx_nfe32]
        # img = Image.open(selected_imgs_nfe32[0])
        # img.save(os.path.join(nfe32_path, f"{0:05}.png"))

        # do nfe20
        # breakpoint()
        sorted_list_nfe20 = sorted(outputs[:20], key=lambda x: f(x))
        topk_scores_nfe20 = sorted_list_nfe20
        topk_idx_nfe20 = [outputs.index(x) for x in topk_scores_nfe20]
        selected_imgs_nfe20 = [imgs[i] for i in topk_idx_nfe20]
        img = Image.open(selected_imgs_nfe20[0])
        img.save(os.path.join(nfe20_path, f"{0:05}.png"))
        # select 4
        for idx in range(4):
            img = Image.open(selected_imgs_nfe20[idx])
            img.save(os.path.join(nfe20_select4_path, f"{idx:05}.png"))


if __name__ == "__main__":
    main()
