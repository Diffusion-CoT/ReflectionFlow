from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from outlines.models.transformers_vision import transformers_vision
from pydantic import BaseModel
import outlines
import torch
from PIL import Image
import os
from typing import Union
import base64
from io import BytesIO

script_dir = os.path.dirname(os.path.abspath(__file__))

import sys

sys.path.append("..")

from utils import load_verifier_prompt


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# Optional device map that one can use to let `transformers` share a single GPU and CPU.
DEVICE_MAP = {
    "visual": 1,
    "model.embed_tokens": 1,
    "model.layers.0": 1,
    "model.layers.1": 1,
    "model.layers.2": 1,
    "model.layers.3": 1,
    "model.layers.4": 1,
    "model.layers.5": 1,
    "model.layers.6": 1,
    "model.layers.7": 1,
    "model.layers.8": 1,
    "model.layers.9": 1,
    "model.layers.10": 1,
    "model.layers.11": "cpu",
    "model.layers.12": "cpu",
    "model.layers.13": "cpu",
    "model.layers.14": "cpu",
    "model.layers.15": "cpu",
    "model.layers.16": "cpu",
    "model.layers.17": "cpu",
    "model.layers.18": "cpu",
    "model.layers.19": "cpu",
    "model.layers.20": "cpu",
    "model.layers.21": "cpu",
    "model.layers.22": "cpu",
    "model.layers.23": "cpu",
    "model.layers.24": "cpu",
    "model.layers.25": "cpu",
    "model.layers.26": "cpu",
    "model.layers.27": "cpu",
    "model.norm": "cpu",
    "model.rotary_emb": "cpu",
    "lm_head": "cpu",
}


class Score(BaseModel):
    explanation: str
    score: float


class Grading(BaseModel):
    accuracy_to_prompt: Score
    creativity_and_originality: Score
    visual_quality_and_realism: Score
    consistency_and_cohesion: Score
    emotional_or_thematic_resonance: Score
    overall_score: Score


class QwenVerifier:
    def __init__(self, seed=1994, use_low_gpu_vram=False):
        model, processor = self.load_verifier()
        self.initmodel = model
        self.initprocessor = processor

        model_kwargs = {"torch_dtype": torch.bfloat16}
        if not use_low_gpu_vram:
            model_kwargs.update({"attn_implementation": "flash_attention_2"})
        else:
            model_kwargs.update({"device_map": "auto"})

        model_kwargs.update({"cache_dir": "/ibex/user/zhaol0c/uniediting/qwen25"})

        self.model = transformers_vision(
            MODEL_ID,
            model_class=model.__class__,
            device="cuda:0" if not use_low_gpu_vram else "cpu",  # hard-code device.
            model_kwargs=model_kwargs,
            processor_class=processor.__class__,
        )
        self.structured_generator = outlines.generate.json(self.model, Grading)
        del model, processor

        self.verifier_prompt = load_verifier_prompt(os.path.join(script_dir, "verifier_prompt.txt"))
        self.seed = seed
        system_instruction_refine = load_verifier_prompt(os.path.join(script_dir, "refine_prompt.txt"))
        system_instruction_reflexion = load_verifier_prompt(os.path.join(script_dir, "reflexion_pompt.txt"))
        self.system_message_refine = {
            "role": "system",
            "content": system_instruction_refine
        }
        self.system_message_reflexion = {
            "role": "system",
            "content": system_instruction_reflexion
        }

    @torch.no_grad()
    def load_verifier(self):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, cache_dir="/ibex/user/zhaol0c/uniediting/qwen25", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map="auto")
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=min_pixels, max_pixels=max_pixels, cache_dir="/ibex/user/zhaol0c/uniediting/qwen25")
        return model, processor

    def prepare_conversations(self, prompt):
        user_content = []
        conversation = [
            {"role": "system", "content": self.verifier_prompt},
        ]
        user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": prompt})
        user_content = {"role": "user", "content": user_content}
        conversation.append(user_content)
        return conversation

    def prepare_inputs(self, images: Union[list[Image.Image], Image.Image], prompts: Union[list[str], str]) -> dict:
        assert len(images) == len(prompts)

        conversations = []
        for prompt in prompts:
            conversations.append(self.prepare_conversations(prompt))

        assert len(conversations) == len(images) == len(prompts)

        prompts = [self.model.processor.apply_chat_template(msg, add_generation_prompt=True) for msg in conversations]
        images = [[image] for image in images]
        inputs = {"images": images, "prompts": prompts}
        return inputs

    @torch.no_grad()
    def score(self, inputs, max_new_tokens) -> list[dict[str, float]]:
        # TODO: might need to iterate `inputs` in batches depending on the resources.
        outputs = self.structured_generator(
            inputs["prompts"], inputs["images"], max_tokens=max_new_tokens, seed=self.seed
        )
        outputs = [o.dict() for o in outputs]
        return outputs

    def prepare_refine_prompt_inputs(self, images: Union[list[Image.Image], Image.Image], 
                                   evaluations: Union[list[str], str], 
                                   original_prompt: Union[list[str], str], 
                                   current_prompt: Union[list[str], str]) -> dict:
        """Prepare inputs for refine prompt (Qwen版本)"""
        inputs = []
        images = images if isinstance(images, list) else [images]
        evaluations = evaluations if isinstance(evaluations, list) else [evaluations]
        original_prompt = original_prompt if isinstance(original_prompt, list) else [original_prompt]
        current_prompt = current_prompt if isinstance(current_prompt, list) else [current_prompt]

        conversations = []
        for img, eval_text, orig_p, curr_p in zip(images, evaluations, original_prompt, current_prompt):
            user_content = [
                {"type": "text", "text": "Original prompt: " + orig_p},
                {"type": "text", "text": "Current prompt: " + curr_p},
                {"type": "text", "text": "Generated images:"},
                {"type": "image"},
                {"type": "text", "text": "Evaluation of the generated images: " + eval_text},
                {"type": "text", "text": "Please refine the current prompt to improve the overall quality of the generated images."}
            ]
            
            conversation = [
                self.system_message_refine,
                {"role": "user", "content": user_content}
            ]

            processed_images = [img.convert("RGB")] if isinstance(img, Image.Image) else [Image.open(img).convert("RGB")]
            
            prompt = self.model.processor.apply_chat_template(conversation, add_generation_prompt=True)
            
            conversations.append({
                "prompt": prompt,
                "images": processed_images
            })

        return {
            "conversations": conversations,
            "original_prompts": original_prompt,
            "current_prompts": current_prompt
        }

    @torch.no_grad()
    def refine_prompt(self, inputs, max_new_tokens=1280) -> list[str]:
        results = []
        for conv in inputs["conversations"]:
            # prepare qwen input
            model_inputs = self.initprocessor(
                text=conv["prompt"],
                images=conv["images"],
                return_tensors="pt"
            ).to("cuda")
            
            # generate
            generated_ids = self.initmodel.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
            )
            
            # decode
            result = self.initprocessor.decode(
                generated_ids[0],
                skip_special_tokens=True
            ).strip()
            results.append(result)
        return results

    def prepare_reflexion_prompt_inputs(self, images: Union[list[Image.Image], Image.Image],
                                      original_prompt: Union[list[str], str],
                                      current_prompt: Union[list[str], str],
                                      reflections: Union[list[str], str]) -> dict:
        conversations = []
        processed_images = []
        
        images = images if isinstance(images, list) else [images]
        original_prompt = original_prompt if isinstance(original_prompt, list) else [original_prompt]
        current_prompt = current_prompt if isinstance(current_prompt, list) else [current_prompt]
        reflections = reflections if isinstance(reflections, list) else [reflections]

        for img, orig_p, curr_p, refl in zip(images, original_prompt, current_prompt, reflections):
            # create conversation content
            input_prompt = f"{curr_p}<Reflection>: {refl}"
            user_content = [
                {"type": "text", "text": "Original prompt: " + orig_p},
                {"type": "text", "text": f"The updated prompt to generate the image is: {input_prompt}"},
                {"type": "text", "text": "Generated images:"},
                {"type": "image"},
                {"type": "text", "text": "Please generate instructions following the defined rules."}
            ]
            
            # create conversation template
            conversation = [
                self.system_message_reflexion,
                {"role": "user", "content": user_content}
            ]
            
            # process image
            processed_img = [img.convert("RGB")] if isinstance(img, Image.Image) else [Image.open(img).convert("RGB")]
            
            prompt = self.model.processor.apply_chat_template(conversation, add_generation_prompt=True)
            
            conversations.append({
                "prompt": prompt,
                "images": processed_img
            })

        return {
            "conversations": conversations,
            "original_prompts": original_prompt,
            "reflections": reflections
        }

    @torch.no_grad()
    def generate_reflections(self, inputs, max_new_tokens=1280) -> list[str]:
        results = []
        for conv in inputs["conversations"]:
            # prepare qwen input
            model_inputs = self.initprocessor(
                text=conv["prompt"],
                images=conv["images"],
                return_tensors="pt"
            ).to("cuda")
            
            # generate
            generated_ids = self.initmodel.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
            )
            
            # decode
            result = self.initprocessor.decode(
                generated_ids[0],
                skip_special_tokens=True
            ).strip()
            results.append(result)
        return results