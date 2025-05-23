import ast
import json
import os
import pdb
import argparse
from collections.abc import Mapping
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('ReflectionFlow')
sys.path.append('reward_modeling')
import torch
from vision_process import process_vision_info

from data import DataConfig
from reward_modeling.utils import ModelConfig, PEFTLoraConfig, TrainingConfig
from reward_modeling.utils import load_model_from_checkpoint
from train_reward import create_model_and_processor
from prompt_template import build_prompt


def load_configs_from_json(config_path):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # del config_dict["training_args"]["_n_gpu"]
    del config_dict["data_config"]["meta_data"]
    del config_dict["data_config"]["data_dir"]

    return config_dict["data_config"], None, config_dict["model_config"], config_dict["peft_lora_config"], \
           config_dict["inference_config"] if "inference_config" in config_dict else None

class ImageVLMRewardInference():
    def __init__(self, load_from_pretrained, load_from_pretrained_step=-1, device='cuda', dtype=torch.bfloat16):
        config_path = os.path.join(load_from_pretrained, "model_config.json")
        data_config, _, model_config, peft_lora_config, inference_config = load_configs_from_json(config_path)
        data_config = DataConfig(**data_config)
        model_config = ModelConfig(**model_config)
        peft_lora_config = PEFTLoraConfig(**peft_lora_config)

        training_args = TrainingConfig(
            load_from_pretrained=load_from_pretrained,
            load_from_pretrained_step=load_from_pretrained_step,
            gradient_checkpointing=False,
            disable_flash_attn2=False,
            bf16=True if dtype == torch.bfloat16 else False,
            fp16=True if dtype == torch.float16 else False,
            output_dir="",
        )
        
        model, processor, peft_config = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=training_args,
        )

        self.device = device

        model, checkpoint_step = load_model_from_checkpoint(model, load_from_pretrained, load_from_pretrained_step)
        model.eval()

        self.model = model
        self.processor = processor

        self.model.to(self.device)

        self.data_config = data_config

        self.inference_config = inference_config

    def _norm(self, reward):
        if self.inference_config is None:
            return reward
        else:
            reward['VQ'] = (reward['VQ'] - self.inference_config['VQ_mean']) / self.inference_config['VQ_std']
            return reward

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side='right'):
        """
        Pad the sequences to the maximum length.
        """
        assert padding_side in ['right', 'left']
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask
        
        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == 'right' else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(sequences, padding, 'constant', self.processor.tokenizer.pad_token_id)
        attention_mask_padded = torch.nn.functional.pad(attention_mask, padding, 'constant', 0)

        return sequences_padded, attention_mask_padded
    
    def _prepare_input(self, data):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            ## TODO: Maybe need to add dtype
            # if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
            #     # NLP models inputs are int/uint and those get adjusted to the right dtype of the
            #     # embedding. Other models such as wav2vec2's inputs are already float and thus
            #     # may need special handling to match the dtypes of the model
            #     kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data
    
    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs
    
    def prepare_batch(self, image_paths, prompts, max_pixels=None,):
        max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels

        chat_data = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image", 
                            "image": image_path, 
                            "max_pixels": max_pixels, 
                        },
                        {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                    ],
                },
            ] for image_path, prompt in zip(image_paths, prompts)
        ]
        image_inputs, video_inputs = process_vision_info(chat_data)

        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self._prepare_inputs(batch)
        
        return batch

    def reward(self, image_paths, prompts, max_pixels=None, use_norm=True):
        """
        Inputs:
            image_paths: List[str], B paths of the videos.
            prompts: List[str], B prompts for the videos.
            eval_dims: List[str], N evaluation dimensions.
            max_pixels: int, maximum pixels of the videos. If None, use the default value in the config.
            use_norm: bool, whether to rescale the output rewards
        Outputs:
            Rewards: List[dict], N + 1 rewards of the B videos.
        """
        
        batch = self.prepare_batch(image_paths, prompts, max_pixels)
        rewards = self.model(
            return_dict=True,
            **batch
        )["logits"]

        rewards = [{'VQ': reward[0].item()} for reward in rewards]
        for i in range(len(rewards)):
            if use_norm:
                rewards[i] = self._norm(rewards[i])
            rewards[i]['Overall'] = rewards[i]['VQ']

        return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Alignment Reward Inference")
    parser.add_argument("--json_path", type=str, default="/mnt/petrelfs/zhuole/gaopeng_for_zl/data/reflection/geneval_pairs.json", 
                        help="Path to input JSON file")
    parser.add_argument("--load_from_pretrained", type=str, default="/mnt/petrelfs/zhuole/VideoAlign/rm_output", 
                        help="Path to pretrained model")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to run inference on")
    parser.add_argument("--output_path", type=str, default="/mnt/petrelfs/zhuole/data/geneval_pairs_reward.json", 
                        help="Path to output JSON file")
    parser.add_argument("--start_index", type=int, default=0, 
                        help="Start index for processing")
    parser.add_argument("--end_index", type=int, default=-1, 
                        help="End index for processing (-1 for all)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing")
    parser.add_argument("--ckpt_step", type=int, default=-1,
                        help="Checkpoint step for processing")
    args = parser.parse_args()
    
    with open(args.json_path, "r") as f:
        data = json.load(f)
    
    # Check if output file exists and load processed items
    if os.path.exists(args.output_path):
        with open(args.output_path, "r") as f:
            outputs = json.load(f)
    else:
        outputs = []
        
    # Process data and check for already processed items
    args.end_index = len(data) if args.end_index == -1 else args.end_index
    processed_count = len(outputs)
    to_process = []
    for idx, item in enumerate(data[args.start_index:args.end_index]):
        item["good_image"] = item["good_image"].replace("gaopeng/zl", "gaopeng_for_zl").replace("ReflectionFlow", "data/reflection")
        item["bad_image"] = item["bad_image"].replace("gaopeng/zl", "gaopeng_for_zl").replace("ReflectionFlow", "data/reflection")
        
        # Skip if already processed
        if idx < processed_count:
            if outputs[idx]["good_image"] == item["good_image"] and \
               outputs[idx]["bad_image"] == item["bad_image"]:
                print(f"Skipping {idx} because it already exists")
                continue
            else:
                raise ValueError(f"Can't find {idx} in outputs")
        
        to_process.append(item)
    
    inferencer = ImageVLMRewardInference(args.load_from_pretrained, load_from_pretrained_step=args.ckpt_step, device=args.device, dtype=torch.bfloat16)
    
    # Process in batches
    for i in tqdm(range(0, len(to_process), args.batch_size), 
                  desc="Processing batches"):
        batch_items = to_process[i:i+args.batch_size]
        
        # Prepare batch data
        good_image_paths = []
        bad_image_paths = []
        good_prompts = []
        bad_prompts = []
        for item in batch_items:
            good_image_paths.append(item["good_image"])
            bad_image_paths.append(item["bad_image"])
            good_prompts.append(item["prompt"])
            bad_prompts.append(item["prompt"])
        image_paths = good_image_paths + bad_image_paths
        prompts = good_prompts + bad_prompts
        
        with torch.no_grad():
            batch_rewards = inferencer.reward(image_paths, prompts, use_norm=True)
            
            good_rewards = batch_rewards[:len(good_image_paths)]
            bad_rewards = batch_rewards[len(good_image_paths):]
            # Process results for each item in the batch
            for j, item in enumerate(batch_items):
                item_with_rewards = item.copy()
                item_with_rewards["good_reward"] = good_rewards[j]["VQ"]
                item_with_rewards["bad_reward"] = bad_rewards[j]["VQ"]
                outputs.append(item_with_rewards)
            
            # Save after each batch to allow resuming
            with open(args.output_path, "w") as f:
                json.dump(outputs, f, indent=4)
