from PIL import Image
import os
import json
import math
import pyarrow.parquet as pq
import numpy as np
import glob
import io
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
from typing import List, Dict

def is_absolute_path(path):
    return os.path.isabs(path)


class ImageConditionDataset(Dataset):
    def __init__(
        self,
        data_path: Dict[str, List[str]] = None,
        condition_size: int = 1024,
        target_size: int = 1024,
        condition_type: str = "cot",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        drop_reflection_prob: float = 0.2,
        return_pil_image: bool = False,
        root_dir: str = "",
        split_ratios: Dict[str, List[float]] = None,
        training_stages: List[int] = None,
    ):
        self.condition_size = condition_size
        self.target_size = target_size
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.drop_reflection_prob = drop_reflection_prob
        self.return_pil_image = return_pil_image
        self.root_dir = root_dir
        self.to_tensor = T.ToTensor()

        # Build dataset for each split.
        self.base_dataset = {}
        for split, paths in data_path.items():
            if not isinstance(paths, list):
                paths = [paths]
            self.base_dataset[split] = []
            for file in paths:
                data = self._load_data_from_file(file)
                self.base_dataset[split].extend(data)
                
        # Set up the training stage and split ratios.
        self.training_stages = training_stages
        self.splits = list(self.base_dataset.keys())
        self.all_split_ratios = split_ratios
        if split_ratios is not None:
            self.split_ratios = {cat: split_ratios[cat][0] for cat in self.splits}
        else:
            self.split_ratios = {cat: 1 / len(self.splits) for cat in self.splits}
    
    def _update_split_ratios(self, current_iter):
        # Check if current iteration is beyond the last stage
        if current_iter >= self.training_stages[-1]:
            # Use the last split ratio directly
            for split in self.splits:
                self.split_ratios[split] = self.all_split_ratios[split][-1]
            # print(f"Using final split ratios: {self.split_ratios} at iteration {current_iter}")
            return
            
        # Find current stage index
        current_stage_idx = 0
        for i, stage_end in enumerate(self.training_stages[1:], 0):
            if current_iter < stage_end:
                break
            current_stage_idx = i
        
        # Calculate stage boundaries
        stage_start = self.training_stages[current_stage_idx]
        stage_end = self.training_stages[current_stage_idx + 1]
        progress = (current_iter - stage_start) / (stage_end - stage_start)
        
        # Update all split ratios at once
        for split in self.splits:
            start_ratio = self.all_split_ratios[split][current_stage_idx]
            end_ratio = self.all_split_ratios[split][current_stage_idx + 1]
            self.split_ratios[split] = start_ratio + progress * (end_ratio - start_ratio)
            
        # print(f"Updated split ratios: {self.split_ratios} at iteration {current_iter}")
        
    def _load_data_from_file(self, file_path):
        """Helper method to load data from a file based on its extension."""
        if file_path.endswith(".json"):
            with open(file_path, "r") as f:
                return json.load(f)
        elif file_path.endswith(".jsonl"):
            data = []
            with open(file_path, "r") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data
        elif file_path.endswith(".parquet"):
            table = pq.read_table(file_path)
            df = table.to_pandas()
            return df.to_dict("records")
        elif os.path.isdir(file_path):
            data = []
            parquet_files = glob.glob(os.path.join(file_path, "*.parquet"))
            print(f"Loading {len(parquet_files)} parquet files from {file_path}")
            # Use ParquetDataset for efficient loading of multiple parquet files
            dataset = pq.ParquetDataset(parquet_files)
            table = dataset.read()
            df = table.to_pandas()
            return df.to_dict("records")
        else:
            raise ValueError(f"Unsupported file type or path: {file_path}")

    def __len__(self):
        # Total length is the sum of samples in all splits.
        return sum(len(samples) for samples in self.base_dataset.values())

    def _sample_split(self):
        """Randomly sample a split based on the provided ratios."""
        return random.choices(self.splits, weights=[self.split_ratios[split] for split in self.splits], k=1)[0]

    def __getitem__(self, idx):
        """Get an item by sampling a split first and then a random item within that split."""
        split = self._sample_split()
        split_data = self.base_dataset[split]
        if not split_data:
            raise ValueError(f"No data available in split: {split}")
        idx = random.randint(0, len(split_data) - 1)
        item = split_data[idx]

        if "good_image" in item:
            good_image_path = os.path.join(self.root_dir, item["good_image"])
            good_image = Image.open(good_image_path)
        elif "edited_img.bytes" in item:
            good_image = Image.open(io.BytesIO(item["edited_img.bytes"]))
        else:
            raise ValueError(f"No good image found in item: {item}")
        if "bad_image" in item:
            bad_image_path = os.path.join(self.root_dir, item["bad_image"])
            bad_image = Image.open(bad_image_path)
        elif "src_img.bytes" in item:
            bad_image = Image.open(io.BytesIO(item["src_img.bytes"]))
        else:
            raise ValueError(f"No good or bad image found in item: {item}")
        
        # Convert images to RGB
        good_image = good_image.convert("RGB")
        bad_image = bad_image.convert("RGB")
        
        # First resize bad image to match good image dimensions to maintain pixel correspondence
        good_w, good_h = good_image.size
        bad_w, bad_h = bad_image.size
        
        # Resize bad image to match good image dimensions
        bad_image = bad_image.resize((good_w, good_h), Image.BICUBIC)
        
        # Now both images have the same dimensions
        # Resize the shorter edge to target_size while maintaining aspect ratio
        ratio = self.target_size / min(good_w, good_h)
        
        new_w = math.ceil(good_w * ratio)
        new_h = math.ceil(good_h * ratio)
        
        # Resize both images to the same dimensions
        good_image = good_image.resize((new_w, new_h), Image.BICUBIC)
        bad_image = bad_image.resize((new_w, new_h), Image.BICUBIC)
        
        # Crop both images to exactly target_size x target_size using the same crop coordinates
        if new_w > self.target_size or new_h > self.target_size:
            left = random.randint(0, new_w - self.target_size)
            top = random.randint(0, new_h - self.target_size)
            
            # Apply the same crop to both images to maintain pixel correspondence
            good_image = good_image.crop((left, top, left + self.target_size, top + self.target_size))
            bad_image = bad_image.crop((left, top, left + self.target_size, top + self.target_size))
        
        # Finally, resize bad_image to condition_size
        bad_image = bad_image.resize((self.condition_size, self.condition_size), Image.BICUBIC)
        
        if "prompt" in item:
            original_prompt = item["prompt"]
        elif "caption" in item:
            original_prompt = item["caption"]
        else:
            raise ValueError(f"No prompt found in item: {item}")
        
        if "reflection_dict" in item:
            # Extract reflection from reflection_dict
            reflection_dict = item["reflection_dict"]
            # Get all available keys in the reflection_dict
            available_keys = list(reflection_dict.keys())
            # Randomly select one key
            random_key = random.choice(available_keys)
            # Get the reflection text from the selected key
            reflection = reflection_dict[random_key]
        elif "instruction" in item:
            reflection = item["instruction"]
        elif "reflection" in item:
            reflection = item["reflection"]
        elif "reflection_prompt" in item:
            reflection = item["reflection_prompt"]
        elif "edited_prompt_list" in item:
            reflection = item["edited_prompt_list"][-1]
        else:
            raise ValueError(f"No reflection found in item: {item}")
        
        # Randomly drop text or image
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        drop_image = drop_image and split != "editing" # don't drop image for editing split
        drop_reflection = random.random() < self.drop_reflection_prob
        drop_reflection = drop_reflection or len(reflection) < 5 # also drop reflection if it's too short
        if drop_reflection or drop_image:
            description = f"{original_prompt}"
        else:
            description = f"{original_prompt} [Reflexion] {reflection}"
        if drop_text:
            description = ""
        if drop_image:
            bad_image = Image.new(
                "RGB", (self.condition_size, self.condition_size), (0, 0, 0)
            )

        return {
            "image": self.to_tensor(good_image),
            "original_prompt": original_prompt,
            "condition": self.to_tensor(bad_image),
            "condition_type": self.condition_type,
            "description": description,
            "position_delta": np.array([0, -self.condition_size // 16]),
            **({"pil_image": [good_image, bad_image]} if self.return_pil_image else {}),
        }