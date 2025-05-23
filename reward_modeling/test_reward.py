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
            cache_dir="/ibex/user/zhaol0c/uniediting_continue/our_reward/initialreward"
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
        assert padding_side in ['right', 'left']
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask
        
        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == 'right' else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(sequences, padding, 'constant', self.processor.tokenizer.pad_token_id)
        attention_mask_padded = torch.nn.functional.pad(attention_mask, padding, 'constant', 0)

        return sequences_padded, attention_mask_padded
    
    def _prepare_input(self, data):
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            return data.to(**kwargs)
        return data
    
    def _prepare_inputs(self, inputs):
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs
    
    def prepare_batch(self, image_paths, prompts, max_pixels=None):
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
    parser.add_argument("--load_from_pretrained", type=str, required=True, 
                        help="Path to pretrained model")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to run inference on")
    parser.add_argument("--ckpt_step", type=int, default=-1,
                        help="Checkpoint step for processing")
    args = parser.parse_args()

    inferencer = ImageVLMRewardInference(args.load_from_pretrained, load_from_pretrained_step=args.ckpt_step, device=args.device, dtype=torch.bfloat16)

    # 手动输入图像路径和 Prompt
    image_paths = ["/ibex/user/zhaol0c/uniediting_continue/nvilaverifier_exps/b2_d16_6000model/00548/samples_best/00001.png", "/ibex/user/zhaol0c/uniediting_continue/nvilaverifier_exps/b2_d16_6000model/00548/midimg/14_round@958442093.png"]
    prompts = ["a photo of a yellow bicycle and a red motorcycle", "a photo of a yellow bicycle and a red motorcycle"]

    # 进行打分
    with torch.no_grad():
        rewards = inferencer.reward(image_paths, prompts, use_norm=True)
        breakpoint()
        print(f"Rewards: {rewards}")