from google import genai
from google.genai import types
import typing_extensions as typing
import json
import os
from typing import Union
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

script_dir = os.path.dirname(os.path.abspath(__file__))

import sys

sys.path.append("..")

from utils import load_verifier_prompt, convert_to_bytes


class Score(typing.TypedDict):
    explanation: str
    score: float


class Grading(typing.TypedDict):
    accuracy_to_prompt: Score
    creativity_and_originality: Score
    visual_quality_and_realism: Score
    consistency_and_cohesion: Score
    emotional_or_thematic_resonance: Score
    overall_score: Score


class GeminiVerifier:
    def __init__(self, seed=1994, model_name="gemini-2.0-flash"):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        system_instruction = load_verifier_prompt(os.path.join(script_dir, "verifier_prompt.txt"))
        self.generation_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=list[Grading],
            seed=seed,
        )
        self.model_name = model_name

    def prepare_inputs(self, images: Union[list[Image.Image], Image.Image], prompts: Union[list[str], str], **kwargs):
        """Prepare inputs for the API from a given prompt and image."""
        inputs = []
        images = images if isinstance(images, list) else [images]
        prompts = prompts if isinstance(prompts, list) else [prompts]
        for prompt, image in zip(prompts, images):
            part = [
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=convert_to_bytes(image), mime_type="image/png"),
            ]
            inputs.extend(part)

        return inputs

    def score(self, inputs, **kwargs) -> list[dict[str, float]]:
        # Group the flat list into consecutive chunks of 2.
        def call_generate_content(parts):
            content = types.Content(parts=parts, role="user")
            response = self.client.models.generate_content(
                model=self.model_name, contents=content, config=self.generation_config
            )
            return response.parsed[0]

        grouped_inputs = [inputs[i : i + 2] for i in range(0, len(inputs), 2)]
        results = []
        max_workers = len(grouped_inputs)
        if max_workers > 4:
            max_workers = 4
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(call_generate_content, group) for group in grouped_inputs]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    # Handle exceptions as appropriate.
                    print(f"An error occurred during API call: {e}")
        return results


# Define inputs
if __name__ == "__main__":
    verifier = GeminiVerifier()
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
        images.append(path_or_url)

    # # Single image
    # response = client.models.generate_content(
    #     model='gemini-2.0-flash',
    #     contents=[
    #         "realistic photo a shiny black SUV car with a mountain in the background.",
    #         load_image("https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/assets/car.jpg")
    #     ],
    #     config=generation_config
    # )
    inputs = verifier.prepare_inputs(images=images, prompts=prompts)
    response = verifier.score(inputs)

    with open("results.json", "w") as f:
        json.dump(response, f)

    print(json.dumps(response, indent=4))
