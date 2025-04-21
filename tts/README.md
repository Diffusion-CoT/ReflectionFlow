# tt-scale-flux

Simple re-implementation of inference-time scaling Flux.1-Dev as introduced in [Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps](https://arxiv.org/abs/2501.09732) by Ma et al. We implement the random search strategy to scale the inference compute budget.

<div align="center">
<img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/tt-scale-flux/collage_Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_i@1-4.png" width=600/>
<p><i>Photo of an athlete cat explaining it’s latest scandal at a press conference to journalists.</i></p>
</div>

**Updates**

🔥 16/02/2025: Support for batched image generation has been added [in this PR](https://github.com/sayakpaul/tt-scale-flux/pull/9). It speeds up the total time but consumes more memory.

🔥 15/02/2025: Support for structured generation with Qwen2.5 has been added (using `outlines` and `pydantic`) in [this PR](https://github.com/sayakpaul/tt-scale-flux/pull/6).

🔥 15/02/2025: Support to load other pipelines has been added in [this PR](https://github.com/sayakpaul/tt-scale-flux/pull/5)! [Result section](#more-results) has been updated, too.

## Getting started

Make sure to install the dependencies: `pip install -r requirements`. The codebase was tested using a single H100 and two H100s (both 80GB variants).

By default, we use [Gemini 2.0 Flash](https://deepmind.google/technologies/gemini/flash/) as the verifier (you [can use](#controlling-the-verifier) Qwen2.5, too). This requires two things:

* `GEMINI_API_KEY` (obtain it from [here](https://ai.google.dev/gemini-api/docs)).
* `google-genai` Python [library](https://pypi.org/project/google-genai/).

Now, fire up:

```bash
GEMINI_API_KEY=... python main.py --prompt="a tiny astronaut hatching from an egg on the moon" --num_prompts=None
```

If you want to use from the [data-is-better-together/open-image-preferences-v1-binarized](https://huggingface.co/datasets/data-is-better-together/open-image-preferences-v1-binarized) dataset, you can just run:

```bash
GEMINI_API_KEY=... python main.py
```

After this is done executing, you should expect a folder named `output` with the following structure:

<details>
<summary>Click to expand</summary>

```bash
output/flux.1-dev/gemini/overall_score/20250215_141308$ tree 
.
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@1_s@1039315023.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@1_s@77559330.json
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@1_s@77559330.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@2_s@1046091514.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@2_s@1388753168.json
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@2_s@1388753168.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@2_s@1527774201.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@2_s@1632020675.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@3_s@1648932110.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@3_s@2033640094.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@3_s@2056028012.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@3_s@510118118.json
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@3_s@510118118.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@3_s@544879571.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@3_s@722867022.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@3_s@951309743.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@3_s@973580742.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@1169137714.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@1271234848.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@1327836930.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@1589777351.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@1592595351.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@1654773907.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@1901647417.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@1916603945.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@209448213.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@2104826872.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@532500803.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@710122236.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@744797903.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@754998363.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@823891989.png
├── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@836183088.json
└── prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@836183088.png
```

</details>

Each JSON file should look like so:

<details>
<summary>Click to expand</summary>

```json
{
    "prompt": "Photo of an athlete cat explaining it\u2019s latest scandal at a press conference to journalists.",
    "search_round": 4,
    "num_noises": 16,
    "best_noise_seed": 1940263961,
    "best_score": {
        "explanation": "The image excels in accuracy, visual quality, and originality, with minor deductions for thematic resonance. Overall, it's a well-executed and imaginative response to the prompt.",
        "score": 9.0
    },
    "choice_of_metric": "overall_score",
    "best_img_path": "output/flux.1-dev/gemini/overall_score/20250216_135414/prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@4_s@1940263961.png"
}
```

</details>

To limit the number of prompts, specify `--num_prompts`. By default, we use 2 prompts. Specify "--num_prompts=all" to use all. 

The output directory should also contain a `config.json`, looking like so:

<details>
<summary>Click to expand</summary>

```json
{
  "max_new_tokens": 300,
  "use_low_gpu_vram": false,
  "choice_of_metric": "overall_score",
  "verifier_to_use": "gemini",
  "torch_dtype": "bf16",
  "height": 1024,
  "width": 1024,
  "max_sequence_length": 512,
  "guidance_scale": 3.5,
  "num_inference_steps": 50,
  "pipeline_config_path": "configs/flux.1_dev.json",
  "search_rounds": 4,
  "prompt": "an anime illustration of a wiener schnitzel",
  "num_prompts": null
}
```

</details>

Once the results are generated, process the results by running:

```bash
python process_results.py --path=path_to_the_output_dir
```

This should output a collage of the best images generated in each search round, grouped by the same prompt.

By default, the `--batch_size_for_img_gen` is set to 1. To speed up the process (at the expense of more memory),
this number can be increased.

## Controlling the pipeline checkpoint and `__call__()` args

This is controlled via the `--pipeline_config_path` CLI args. By default, it uses [`configs/flux.1_dev.json`](./configs/flux.1_dev.json). You can either modify this one or create your own JSON file to experiment with different pipelines. We provide some predefined configs for Flux.1-Dev, PixArt-Sigma, SDXL, and SD v1.5 in the [`configs`](./conf) directory.

The above-mentioned pipelines are already supported. To add your own, you need to make modifications to:

* [`MODEL_NAME_MAP`](https://github.com/sayakpaul/tt-scale-flux/blob/8e4ba232fbdfeb7a6879049d3b5765f81969ddf3/utils.py#L16)
* [`get_latent_prep_fn()`](https://github.com/sayakpaul/tt-scale-flux/blob/8e4ba232fbdfeb7a6879049d3b5765f81969ddf3/utils.py#L125C5-L125C23)

## Controlling the "scale"

By default, we use 4 `search_rounds` and start with a noise pool size of 2. Each search round scales up the pool size like so: `2 ** current_seach_round` (with indexing starting from 1). This is where the "scale" in inference-time scaling comes from. You can increase the compute budget by specifying a larger `search_rounds`.

For each search round, we serialize the images and best datapoint (characterized by the best eval score) in a JSON file.

For other supported CLI args, run `python main.py -h`.

## Controlling the verifier

If you don't want to use Gemini, you can use [Qwen2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) as an option. Simply specify `--verifier_to_use=qwen` for this. Below is a
complete command that uses SDXL-base:

```bash
python main.py --verifier_to_use="qwen" --pipeline_config_path=configs/sdxl.json --prompt="Photo of an athlete cat explaining it’s latest scandal at a press conference to journalists." --num_prompts=None --search_rounds=6 --max_new_tokens=800
```

<details>
<summary>Sample search JSON</summary>

```json
{
    "prompt": "Photo of an athlete cat explaining it\u2019s latest scandal at a press conference to journalists.",
    "search_round": 6,
    "num_noises": 64,
    "best_noise_seed": 1937268448,
    "best_score": {
        "explanation": "Overall, the image demonstrates a high level of accuracy, creativity, and theme consistency while maintaining a high visual quality and coherence within the depicted scenario. The humor and surprise value are significant, contributing to above-average scoring.",
        "score": 9.0
    },
    "choice_of_metric": "overall_score",
    "best_img_path": "output/sdxl-base/qwen/overall_score/20250216_141140/prompt@Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_hash@b9094b65_i@6_s@1937268448.png"
}
```

</details>&nbsp;&nbsp;

<details>
<summary>Results</summary>

<table>
  <tr>
    <th>Result</th>
  </tr>
  <tr>
    <td>
      <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/tt-scale-flux/sdxl_qwen_collage_Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_i%401-6.jpeg" alt="scandal_cat" width="650">
      <br>
      <i>Photo of an athlete cat explaining it’s latest scandal at a press conference to journalists.</i>
    </td>
  </tr>
</table>

</details>&nbsp;&nbsp;

> [!IMPORTANT]  
> This setup was tested on 2 H100s. If you want to do this on a single GPU, specify `--use_low_gpu_vram`.

You can also bring in your own verifier by implementing a so-called `Verifier` class following the structure of either of `GeminiVerifier` or `QwenVerifier`. You will then have to make adjustments to the following places:

* [Scoring](https://github.com/sayakpaul/tt-scale-flux/blob/c654bc066171aee9c765fa42a322f65415529a77/main.py#L135)
* [Sorting](https://github.com/sayakpaul/tt-scale-flux/blob/c654bc066171aee9c765fa42a322f65415529a77/main.py#L163)

By default, we use "overall_score" as the metric to obtain the best samples in each search round. You can change it by specifying `--choice_of_metric`. Supported values are: 

* "accuracy_to_prompt"
* "creativity_and_originality"
* "visual_quality_and_realism"
* "consistency_and_cohesion"
* "emotional_or_thematic_resonance"
* "overall_score"

If you're experimenting with a new verifier, you can relax these choices.

The verifier prompt that is used during grading/verification is specified in [this file](./verifiers/verifier_prompt.txt). The prompt is a slightly modified version of the one specified in the Figure 16 of
the paper (Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps). You are welcome to 
experiment with a different prompt.

## More results

<details>
<summary>Click to expand</summary>

<table>
  <tr>
    <th>Result</th>
  </tr>
  <tr>
    <td>
      <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/tt-scale-flux/collage_a_bustling_manga_street_devoid_of_vehicles_detailed_with_vibrant_colors_and_dynamic_l_i@1-4.jpeg" alt="Manga" width="650">
      <br>
      <i>a bustling manga street, devoid of vehicles, detailed with vibrant colors and dynamic<br> line work, characters in the background adding life and movement, under a soft golden<br> hour light, with rich textures and a lively atmosphere, high resolution, sharp focus</i>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/tt-scale-flux/collage_Alice_in_a_vibrant_dreamlike_digital_painting_inside_the_Nemo_Nautilus_submarine__i@1-4.jpeg" alt="Alice" width="650">
      <br>
      <i>Alice in a vibrant, dreamlike digital painting inside the Nemo Nautilus submarine.</i>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/tt-scale-flux/flux_collage_an_anime_illustration_of_a_wiener_schnitzel_i%401-4.png" alt="wiener_schnitzel" width="650">
      <br>
      <i>an anime illustration of a wiener schnitzel</i>
    </td>
  </tr>
</table>

</details>&nbsp;&nbsp;

Both searches were performed with "overall_score" as the metric. Below is example, presenting a comparison
between the outputs of different metrics -- "overall_score" vs. "emotional_or_thematic_resonance" for the prompt:
"a tiny astronaut hatching from an egg on the moon":

<details>
<summary>Click to expand</summary>

<table>
  <tr>
    <th>Metric</th>
    <th>Result</th>
  </tr>
  <tr>
    <td>"overall_score"</td>
    <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/tt-scale-flux/collage_a_tiny_astronaut_hatching_from_an_egg_on_the_moon_i@1-4.png" alt="overall" width="350"></td>
  </tr>
  <tr>
    <td>"emotional_or_thematic_resonance"</td>
    <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/tt-scale-flux/collage_a_tiny_astronaut_hatching_from_an_egg_on_the_moon_i@1-4_thematic.png" alt="Alicet" width="350"></td>
  </tr>
</table>

</details>

### Results from other models

<details>
<summary>PixArt-Sigma</summary>

<table>
  <tr>
    <th>Result</th>
  </tr>
  <tr>
    <td>
      <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/tt-scale-flux/pixart_collage_A_person_playing_saxophone__i%401-4.png" alt="saxophone" width="650">
      <br>
      <i>A person playing saxophone.</i>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/tt-scale-flux/pixart_collage_Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_i%401-4.jpeg" alt="scandal_cat" width="650">
      <br>
      <i>Photo of an athlete cat explaining it’s latest scandal at a press conference to journalists.</i>
    </td>
  </tr>
</table>

</details><br>

<details>
<summary>SD v1.5</summary>

<table>
  <tr>
    <th>Result</th>
  </tr>
  <tr>
    <td>
      <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/tt-scale-flux/sd_collage_a_photo_of_an_astronaut_riding_a_horse_on_mars_i%401-6.png" alt="saxophone" width="650">
      <br>
      <i>a photo of an astronaut riding a horse on mars</i>
    </td>
  </tr>
</table>

</details>&nbsp;&nbsp;

<details>
<summary>SDXL-base</summary>

<table>
  <tr>
    <th>Result</th>
  </tr>
  <tr>
    <td>
      <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/tt-scale-flux/sdxl_collage_Photo_of_an_athlete_cat_explaining_it_s_latest_scandal_at_a_press_conference_to_journ_i%401-6.jpeg" alt="scandal_cat" width="650">
      <br>
      <i>Photo of an athlete cat explaining it’s latest scandal at a press conference to journalists.</i>
    </td>
  </tr>
</table>

</details>&nbsp;&nbsp;

## Acknowledgements

* Thanks to [Willis Ma](https://twitter.com/ma_nanye) for all the guidance and pair-coding.
* Thanks to Hugging Face for supporting the compute.
* Thanks to Google for providing Gemini credits.
* Thanks a bunch to [amitness](https://github.com/amitness) for [this PR](https://github.com/sayakpaul/tt-scale-flux/pull/7).
