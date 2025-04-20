from pydantic import BaseModel
import base64
from io import BytesIO
from openai import OpenAI
import json
import os
from typing import Union
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from diffusers.utils import load_image
import requests
import time

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

class Grading_geneval(BaseModel):
    single_object: Score
    two_object: Score
    counting: Score
    colors: Score
    position: Score
    color_attr: Score
    overall_score: Score

class Grading_single_object(BaseModel):
    object_completeness: Score
    detectability: Score
    occlusion_handling: Score
    overall_score: Score

class Grading_two_object(BaseModel):
    separation_clarity: Score
    individual_completeness: Score
    relationship_accuracy: Score
    overall_score: Score

class Grading_counting(BaseModel):
    count_accuracy: Score
    object_uniformity: Score
    spatial_legibility: Score
    overall_score: Score

class Grading_colors(BaseModel):
    color_fidelity: Score
    contrast_effectiveness: Score
    multi_object_consistency: Score
    overall_score: Score

class Grading_position(BaseModel):
    position_accuracy: Score
    occlusion_management: Score
    perspective_consistency: Score
    overall_score: Score

class Grading_color_attr(BaseModel):
    attribute_binding: Score
    contrast_effectiveness: Score
    material_consistency: Score
    overall_score: Score

class OpenAIVerifier:
    def __init__(self, seed=1994, model_name="gpt-4o-2024-11-20", refine_prompt_relpath="refine_prompt.txt", reflexion_prompt_relpath="reflexion_prompt.txt", verifier_prompt_relpath="verifier_prompt.txt"):
        self.client = OpenAI(
            api_key=os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        )
        self.api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = "https://boyuerichdata.chatgptten.com/v1/chat/completions"
        # self.client = OpenAI(
        #     base_url="https://boyuerichdata.chatgptten.com/v1/",
        #     api_key=os.getenv("API_KEY")
        # )
        self.system_instruction = load_verifier_prompt(os.path.join(script_dir, verifier_prompt_relpath))
        system_instruction_refine = load_verifier_prompt(os.path.join(script_dir, refine_prompt_relpath))
        system_instruction_reflexion = load_verifier_prompt(os.path.join(script_dir, reflexion_prompt_relpath))
        self.system_message_refine = {
            "role": "system",
            "content": system_instruction_refine
        }
        self.system_message_reflexion = {
            "role": "system",
            "content": system_instruction_reflexion
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

    def score(self, inputs, tag=None, **kwargs) -> list[dict[str, float]]:
        if tag == None:
            # general version
            system_message = {
                "role": "system",
                "content": self.system_instruction
            }
            response_format = Grading
        else:
            system_message = {
                "role": "system",
                "content": self.system_instruction[tag]
            }
            if tag == "single_object":
                response_format = Grading_single_object
            elif tag == "two_object":
                response_format = Grading_two_object
            elif tag == "counting":
                response_format = Grading_counting
            elif tag == "colors":
                response_format = Grading_colors
            elif tag == "position":
                response_format = Grading_position
            elif tag == "color_attr":
                response_format = Grading_color_attr
        def call_generate_content(parts):
            conversation = [system_message, parts]
            response = self.client.beta.chat.completions.parse(
                model=self.model_name, messages=conversation, temperature=1, response_format=response_format
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
    
    def prepare_refine_prompt_inputs(
        self,
        original_prompt: Union[list[str], str],
        images: Union[list[Image.Image], Image.Image] = None,
        evaluations: Union[list[str], str] = None,
        current_prompt: Union[list[str], str] = None,
        reflections: Union[list[str], str] = None,
        **kwargs
    ):
        inputs = []

        if original_prompt is None:
            raise TypeError("original_prompt cannot be None")

        def _ensure_list(input_val):
            if input_val is None:
                return [None] * (len(original_prompt) if isinstance(original_prompt, list) else 1)
            return input_val if isinstance(input_val, list) else [input_val]

        original_prompt = _ensure_list(original_prompt)
        images = _ensure_list(images)
        evaluations = _ensure_list(evaluations)
        current_prompt = _ensure_list(current_prompt)
        reflections = _ensure_list(reflections)

        bsz = len(original_prompt)

        # Ensure all input lists are of the same length (batch size) - important for zipping correctly
        images = images[:bsz] if images is not None else [None] * bsz
        evaluations = evaluations[:bsz] if evaluations is not None else [None] * bsz
        current_prompt = current_prompt[:bsz] if current_prompt is not None else [None] * bsz
        reflections = reflections[:bsz] if reflections is not None else [None] * bsz

        for orig_prompt, img, eval_text, curr_prompt, reflection in zip(original_prompt, images, evaluations, current_prompt, reflections):
            message_content = []
            message_content.append({"type": "text", "text": f"Original prompt: {orig_prompt}"})

            if curr_prompt:
                message_content.append({"type": "text", "text": f"Current prompt: {curr_prompt}"})
            if reflection:
                message_content.append({"type": "text", "text": f"Reflection prompt: {reflection}"})

            if img:
                base64_image = None
                try:
                    if isinstance(img, str):
                        with open(img, "rb") as image_file:
                            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    elif isinstance(img, Image.Image):  # If image is a PIL Image
                        buffered = BytesIO()
                        img.save(buffered, format="JPEG")
                        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    else:
                        print(f"Warning: Unexpected image type: {type(img)}. Skipping image.")
                except Exception as e:
                    print(f"Error processing image: {e}. Skipping image.")
                    base64_image = None

                if base64_image:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })

            if eval_text:
                message_content.append({"type": "text", "text": f"Evaluation of the generated images: {eval_text}"})

            message_content.append({"type": "text", "text": "Please refine the current prompt to improve the overall quality of the future generated images."})

            inputs.append({"role": "user", "content": message_content})

        return inputs
    
    # def refine_prompt(self, inputs, **kwargs) -> list[dict[str, float]]:
    #     max_retries=10
    #     retry_delay=5
    #     def call_generate_content(parts):
    #         headers = {
    #             "Content-Type": "application/json",
    #             "Authorization": f"Bearer {self.api_key}"
    #         }
    #         conversation = [self.system_message_refine, parts]
    #         payload = {
    #             "model": self.model_name,
    #             "messages": conversation,
    #             "temperature": 0.0
    #         }
    #         for attempt in range(max_retries):
    #             try:
    #                 response = requests.post(self.base_url, headers=headers, json=payload)
    #                 # breakpoint()
    #                 if response.status_code == 200:
    #                     result_json = response.json()
    #                     return result_json['choices'][0]['message']['content']
    #                 else:
    #                     print(f"API error (attempt {attempt+1}/{max_retries}): Status {response.status_code}")
    #                     if attempt < max_retries - 1:
    #                         time.sleep(retry_delay)
    #             except Exception as e:
    #                 print(f"Request error (attempt {attempt+1}/{max_retries}): {str(e)}")
    #                 if attempt < max_retries - 1:
    #                     time.sleep(retry_delay)
    #         return None
        
    #     results = []
    #     max_workers = min(len(inputs), 4)
    #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #         futures = [executor.submit(call_generate_content, group) for group in inputs]
    #         for future in as_completed(futures):
    #             try:
    #                 results.append(future.result())
    #             except Exception as e:
    #                 # Handle exceptions as appropriate.
    #                 print(f"An error occurred during API call: {e}")
    #     return results

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
    
    def prepare_reflexion_prompt_inputs(self, images: Union[list[Image.Image], Image.Image], original_prompt: Union[list[str], str], current_prompt: Union[list[str], str], reflections: Union[list[str], str], evaluations: Union[list[str], str], **kwargs):
        """Prepare inputs for the API from a given prompt and image."""
        inputs = []
        images = images if isinstance(images, list) else [images]
        original_prompt = original_prompt if isinstance(original_prompt, list) else [original_prompt]
        current_prompt = current_prompt if isinstance(current_prompt, list) else [current_prompt]
        reflections = reflections if isinstance(reflections, list) else [reflections]
        evaluations = evaluations if isinstance(evaluations, list) else [evaluations]
        for image, original_prompt, current_prompt, reflection, evaluation in zip(images, original_prompt, current_prompt, reflections, evaluations):
            # Convert image to base64
            if isinstance(image, str):  # If image is a file path
                with open(image, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            else:  # If image is a PIL Image
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
            input_prompt = current_prompt + "[Reflexion]: " + reflection
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Original prompt: " + original_prompt},
                    {"type": "text", "text": f"The updated prompt to generate the image is: {input_prompt}"},
                    {"type": "text", "text": f"Evaluation of the generated image: {evaluation}"},
                    {"type": "text", "text": "Generated images:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {"type": "text", "text": "Please generate instructions following the defined rules."}
                ]
            }
            inputs.append(message)

        return inputs
    
    def generate_reflections(self, inputs, **kwargs) -> list[dict[str, float]]:
        def call_generate_content(parts):
            conversation = [self.system_message_reflexion, parts]
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
            "realistic photo a shiny black SUV car with a mountain in the background.",
            "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/assets/car.jpg",
        ),
        (
            "photo a green and funny creature standing in front a lightweight forest.",
            "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/assets/green_creature.jpg",
        ),
    ]

    prompts = []
    images = []
    for text, path_or_url in image_urls:
        prompts.append(text)
        images.append(load_image(path_or_url))

    inputs = verifier.prepare_inputs(images=images, prompts=prompts)
    response = verifier.score(inputs)
    
    with open("results.json", "w") as f:
        json.dump(response, f, indent=4)
    
    print(json.dumps(response, indent=4))
    
    # test refine prompt
    # convert response into string
    evaluations = json.dumps(response[0])
    input_data = {
        "images": load_image(image_urls[0][-1]),
        "original_prompt": image_urls[0][0],
        "current_prompt": image_urls[0][0],
        "evaluations": evaluations
    }

    argument_keys_sequence = [
        ["original_prompt"],
        ["original_prompt", "images"],
        ["original_prompt", "images", "current_prompt"],
        ["original_prompt", "images", "current_prompt", "evaluations"]
    ]

    for keys in argument_keys_sequence:
        prepare_args = {key: input_data[key] for key in keys}
        prepared_input = verifier.prepare_refine_prompt_inputs(**prepare_args)
        response = verifier.refine_prompt(prepared_input)
        print(response)
        print("\n")

