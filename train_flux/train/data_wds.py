import math
import random
import glob
import numpy as np
from PIL import Image

import webdataset as wds
from torch.utils.data import IterableDataset, DataLoader
import torchvision.transforms as T

class ImageConditionWebDataset(IterableDataset):
    def __init__(
        self,
        shards_pattern: str,
        condition_size: int = 1024,
        target_size: int = 1024,
        condition_type: str = "cot",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        drop_reflection_prob: float = 0.2,
        split_ratios: dict = None,            # e.g. {"general":[.7,.3], "length":[.3,.7], …}
        training_stages: list = None,         # e.g. [0, 10000, 20000]
        return_pil_image: bool = False,
        shuffle_buffer: int = 1000,
    ):
        super().__init__()
        self.condition_size       = condition_size
        self.target_size          = target_size
        self.condition_type       = condition_type
        self.drop_text_prob       = drop_text_prob
        self.drop_image_prob      = drop_image_prob
        self.drop_reflection_prob = drop_reflection_prob
        self.return_pil_image     = return_pil_image

        # prepare WebDataset pipelines per subset
        self.splits = list(split_ratios.keys())
        self.all_split_ratios = split_ratios
        # start with first stage’s ratios
        self.split_ratios = {s: split_ratios[s][0] for s in self.splits}
        self.training_stages = training_stages or [0]

        # one independent pipeline for each subset
        self.datasets = {}
        if "https://" not in shards_pattern:
            shards_pattern = glob.glob(shards_pattern) 
        for split in self.splits:
            ds = (
                wds.WebDataset(shards_pattern, handler=wds.ignore_and_continue)
                  .shuffle(shuffle_buffer)
                  .decode("pil")  # good_image.jpg / bad_image.jpg → PIL
                  .to_tuple(
                      "good_image.jpg",
                      "bad_image.jpg",
                      "reflection.txt",
                      "prompt.txt",
                      "subset.txt",
                  )
                  # keep only records whose subset matches this split
                  .select(lambda sample: sample[4] == split)
            )
            self.datasets[split] = ds

        # create one iterator per subset
        self.iters = {s: iter(ds) for s, ds in self.datasets.items()}
        self.to_tensor = T.ToTensor()
        self.iteration = 0

    def _update_split_ratios(self):
        itr = self.iteration
        stages = self.training_stages
        # beyond last => use last ratios
        if itr >= stages[-1]:
            for s in self.splits:
                self.split_ratios[s] = self.all_split_ratios[s][-1]
            return

        # find current stage index
        idx = max(i for i, t in enumerate(stages) if itr >= t)
        next_idx = min(idx+1, len(stages)-1)
        start, end = stages[idx], stages[next_idx]
        progress = (itr - start) / (end - start) if end>start else 1.0

        for s in self.splits:
            r0 = self.all_split_ratios[s][idx]
            r1 = self.all_split_ratios[s][next_idx]
            self.split_ratios[s] = r0 + progress*(r1-r0)

    def _preprocess_pair(self, good: Image.Image, bad: Image.Image):
        # match bad → good dims
        gw, gh = good.size
        bad = bad.resize((gw, gh), Image.BICUBIC)
        # scale shorter edge → target_size
        ratio = self.target_size / min(gw, gh)
        nw, nh = math.ceil(gw*ratio), math.ceil(gh*ratio)
        good = good.resize((nw, nh), Image.BICUBIC)
        bad  = bad.resize((nw, nh), Image.BICUBIC)

        # same random crop
        if nw>self.target_size or nh>self.target_size:
            left = random.randint(0, nw-self.target_size)
            top  = random.randint(0, nh-self.target_size)
            box  = (left, top, left+self.target_size, top+self.target_size)
            good = good.crop(box)
            bad  = bad.crop(box)

        # final resize bad → condition_size
        bad = bad.resize((self.condition_size, self.condition_size), Image.BICUBIC)
        return good, bad

    def __iter__(self):
        while True:
            # 1) update dynamic ratios
            self._update_split_ratios()

            # 2) pick a split by current weights
            split = random.choices(
                self.splits,
                weights=[self.split_ratios[s] for s in self.splits],
                k=1,
            )[0]

            # 3) pull next sample (re‑reset iterator on exhaustion)
            try:
                good, bad, ref_bytes, prom_bytes, sub_bytes = next(self.iters[split])
            except StopIteration:
                self.iters[split] = iter(self.datasets[split])
                good, bad, ref_bytes, prom_bytes, sub_bytes = next(self.iters[split])

            # decode text
            reflection = ref_bytes
            prompt     = prom_bytes
            subset     = sub_bytes

            # convert to RGB
            good = good.convert("RGB")
            bad  = bad.convert("RGB")

            # 4) apply your resize/crop logic
            good, bad = self._preprocess_pair(good, bad)

            # 5) decide drops
            drop_text       = random.random() < self.drop_text_prob
            drop_image_flag = random.random() < self.drop_image_prob and subset!="editing"
            drop_reflection = (
                random.random() < self.drop_reflection_prob
                or len(reflection)<5
            )

            if drop_reflection or drop_image_flag:
                description = prompt
            else:
                description = f"{prompt} [Reflexion] {reflection}"
            if drop_text:
                description = ""
            if drop_image_flag:
                # black out condition
                bad = Image.new("RGB", (self.condition_size, self.condition_size), (0,0,0))

            # 6) to tensors
            image     = self.to_tensor(good)
            condition = self.to_tensor(bad)

            out = {
                "image":           image,
                "condition":       condition,
                "original_prompt": prompt,
                "condition_type":  self.condition_type,
                "description":     description,
                "position_delta":  np.array([0, -self.condition_size//16]),
                "subset": subset
            }
            if self.return_pil_image:
                out["pil_image"] = [good, bad]

            self.iteration += 1
            yield out

# usage:
if __name__ == "__main__":

    split_ratios = {
        "general": [0.8, 0.6, 0.4],
        "length":  [0.1, 0.2, 0.3],
        "rule":    [0.1, 0.2, 0.3],
        "editing": [0.0, 0.0, 0.0],
    }
    training_stages = [0, 10000, 20000]

    # local path should be provided as 
    # shards_pattern = "DIR_WHERE_GenRef-wds_IS_DOWNLOADED/*.tar"
    shards_pattern = "pipe:curl -s -f -L https://huggingface.co/datasets/diffusion-cot/GenRef-wds/resolve/main/genref_{0..2}.tar"
    dataset = ImageConditionWebDataset(
        shards_pattern=shards_pattern,
        condition_size=1024,
        target_size=1024,
        condition_type="cot",
        drop_text_prob=0.1,
        drop_image_prob=0.1,
        drop_reflection_prob=0.2,
        split_ratios=split_ratios,
        training_stages=training_stages,
        return_pil_image=False,
    )

    loader = DataLoader(dataset, batch_size=8, num_workers=4)

    # iterate:
    for batch in loader:
        print(batch.keys())
        print(batch["image"].size())
        print(batch["condition"].size())
        break
