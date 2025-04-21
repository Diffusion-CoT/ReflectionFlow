from pydantic import BaseModel
import base64
from io import BytesIO
from openai import OpenAI
import json
import os
from typing import Union
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

script_dir = os.path.dirname(os.path.abspath(__file__))

import sys

sys.path.append("..")

from utils import load_verifier_prompt, convert_to_bytes


class Score(BaseModel):
    score: int
    explanation: str

class Grading(BaseModel):
    accuracy_to_prompt: Score
    creativity_and_originality: Score
    visual_quality_and_realism: Score
    consistency_and_cohesion: Score
    emotional_or_thematic_resonance: Score
    overall_score: Score

class OpenAIVerifier:
    def __init__(self, seed=1994, model_name="gpt-4o-2024-11-20"):
        self.client = OpenAI(
            api_key=os.getenv("API_KEY")
        )
        system_instruction = load_verifier_prompt(os.path.join(script_dir, "verifier_prompt.txt"))
        system_instruction_refine = load_verifier_prompt(os.path.join(script_dir, "refine_prompt.txt"))
        self.system_message = {
            "role": "system",
            "content": system_instruction
        }
        self.system_message_refine = {
            "role": "system",
            "content": system_instruction_refine
        }
        self.model_name = model_name
        self.seed = seed

    def prepare_inputs(self, images: Union[list[Image.Image], Image.Image], prompts: Union[list[str], str], **kwargs):
        """Prepare inputs for the API from a given prompt and image."""
        inputs = []
        images = images if isinstance(images, list) else [images]
        prompts = prompts if isinstance(prompts, list) else [prompts]
        for prompt, image in zip(prompts, images):
            # Convert image to base64
            if isinstance(image, str):  # If image is a file path
                with open(image, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            else:  # If image is a PIL Image
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
            inputs.append(message)

        return inputs

    def score(self, inputs, **kwargs) -> list[dict[str, float]]:
        def call_generate_content(parts):
            conversation = [self.system_message, parts]
            response = self.client.beta.chat.completions.parse(
                model=self.model_name, messages=conversation, temperature=1, response_format=Grading
            )
            return response.choices[0].message.parsed.model_dump()
        
        results = []
        max_workers = min(len(inputs), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(call_generate_content, group) for group in inputs]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    # Handle exceptions as appropriate.
                    print(f"An error occurred during API call: {e}")
        return results
    
    def prepare_refine_prompt_inputs(self, images: Union[list[Image.Image], Image.Image], evaluations: Union[list[str], str], user_prompt: Union[list[str], str], current_prompt: Union[list[str], str], **kwargs):
        """Prepare inputs for the API from a given prompt and image."""
        inputs = []
        images = images if isinstance(images, list) else [images]
        evaluations = evaluations if isinstance(evaluations, list) else [evaluations]
        user_prompt = user_prompt if isinstance(user_prompt, list) else [user_prompt]
        current_prompt = current_prompt if isinstance(current_prompt, list) else [current_prompt]
        for prompt, image, user_prompt, current_prompt in zip(evaluations, images, user_prompt, current_prompt):
            # Convert image to base64
            if isinstance(image, str):  # If image is a file path
                with open(image, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            else:  # If image is a PIL Image
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "User prompt: " + user_prompt},
                    {"type": "text", "text": "Current prompt: " + current_prompt},
                    {"type": "text", "text": "Generated images:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {"type": "text", "text": "Evaluation of the generated images: " + prompt}
                ]
            }
            inputs.append(message)

        return inputs

    def refine_prompt(self, inputs, **kwargs) -> list[dict[str, float]]:
        def call_generate_content(parts):
            conversation = [self.system_message_refine, parts]
            response = self.client.chat.completions.create(
                model=self.model_name, messages=conversation, temperature=1,
            )
            return response.choices[0].message.content
        
        results = []
        max_workers = min(len(inputs), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(call_generate_content, group) for group in inputs]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    # Handle exceptions as appropriate.
                    print(f"An error occurred during API call: {e}")
        return results

# Define inputs
if __name__ == "__main__":
    verifier = OpenAIVerifier()
    image_urls = [
        (
            "a 3D model of a minimalist, abstract financial landscape, featuring floating geometric shapes and volumetric charts with smooth gradients, representing the concept of investing in private equity, bathed in soft golden hour light, with a serene mood and high resolution, no people present",
            "/mnt/petrelfs/gaopeng/zl/data/open_image_preferences_v1_binarized/bad/4-quality.jpg",
        ),
        (
            "Tempera painting of a fragmented, distorted scene from the past, with blurred fragments and incomplete abstract memory. Focus on the eyes with a fearsome appearance, intricate details, and masterful touch, under a dramatic chiaroscuro lighting, rich textures, and a monochromatic scheme.",
            "/mnt/petrelfs/gaopeng/zl/data/open_image_preferences_v1_binarized/bad/15-quality.jpg",
        ),
    ]

    prompts = []
    images = []
    for text, path_or_url in image_urls:
        prompts.append(text)
        images.append(path_or_url)

    inputs = verifier.prepare_inputs(images=images, prompts=prompts)
    response = verifier.score(inputs)
    
    with open("results.json", "w") as f:
        json.dump(response, f, indent=4)
    
    print(json.dumps(response, indent=4))
    
    # test refine prompt
    # convert response into string
    evaluations = json.dumps(response[0])
    input_data = {
        "images": "/mnt/petrelfs/gaopeng/zl/data/open_image_preferences_v1_binarized/bad/15-quality.jpg",
        "user_prompt": "a 3D model of a minimalist, abstract financial landscape, featuring floating geometric shapes and volumetric charts with smooth gradients, representing the concept of investing in private equity, bathed in soft golden hour light, with a serene mood and high resolution, no people present",
        "current_prompt": "a 3D model of a minimalist, abstract financial landscape, featuring floating geometric shapes and volumetric charts with smooth gradients, representing the concept of investing in private equity, bathed in soft golden hour light, with a serene mood and high resolution, no people present",
        "evaluations": evaluations
    }
    inputs = verifier.prepare_refine_prompt_inputs(images=input_data["images"], evaluations=input_data["evaluations"], user_prompt=input_data["user_prompt"], current_prompt=input_data["current_prompt"])
    response = verifier.refine_prompt(inputs)
    print(response)
