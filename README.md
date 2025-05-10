<div align="center" style="font-family: charter;">
<h1><i>From Reflection to Perfection:</i></br>Scaling Inference-Time Optimization for Text-to-Image Diffusion Models via Reflection Tuning</h1>



<a href="https://arxiv.org/abs/2504.16080" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-ReflectionFlow-red?logo=arxiv" height="20" /></a>
<a href="https://diffusion-cot.github.io/reflection2perfection/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-ReflectionFlow-blue.svg" height="20" /></a>
<a href="https://huggingface.co/collections/diffusion-cot/reflectionflow-release-6803e14352b1b13a16aeda44" target="_blank">
    <img alt="HF Dataset: ReflectionFlow" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Hugging Face-ReflectionFlow-ffc107?color=ffc107&logoColor=white" height="20" /></a>

<div>
    <a href="https://le-zhuo.com/" target="_blank">Le Zhuo</a><sup>1,</sup><sup>4</sup>,</span>
    <a href="https://liangbingzhao.github.io/" target="_blank">Liangbing Zhao</a><sup>2</sup>, </span>
    <a href="https://sayak.dev/" target="_blank">Sayak Paul</a><sup>3</sup>,</span>
    <a href="https://liaoyue.net" target="_blank">Yue Liao</a><sup>1</sup>,</span>
    <a href="https://zrrskywalker.github.io/" target="_blank">Renrui Zhang</a><sup>1</sup>,</span>
    <a href="https://synbol.github.io/" target="_blank">Yi Xin</a><sup>4</sup>,</span>
    <a href="https://gaopengcuhk.github.io/" target="_blank">Peng Gao</a><sup>4</sup>,</span>
    <br>
    <a href="https://cemse.kaust.edu.sa/profiles/mohamed-elhoseiny" target="_blank">Mohamed Elhoseiny</a><sup>2</sup>, </span>
    <a href="https://www.ee.cuhk.edu.hk/~hsli/" target="_blank">Hongsheng Li</a><sup>1</sup></span>
</div>


<div>
    <sup>1</sup>CUHK MMLAB&emsp;
    <sup>2</sup>KAUST&emsp;
    <sup>3</sup>Hugging Face&emsp;
    <sup>4</sup>Shanghai AI Lab&emsp;
</div>


<img src="examples/teaser.jpg" width="100%"/>

<h align="justify">Overall pipeline of the <strong>ReflectionFlow</strong> framework with qualitative and quantitative results of scaling compute at inference time.</h>

</div>      

## :fire: News

- [2025/4/23] Release [paper](https://arxiv.org/abs/2504.16080).
- [2025/4/20] Release GenRef dataset, model checkpoints, as well as the training and inference code.

## ✨ Quick Start  

### Installation

1. **Environment setup**
```bash
conda create -n ReflectionFlow python=3.10
conda activate ReflectionFlow
```
2. **Requirements installation**
```bash
pip install -r requirements.txt
```

## 🚀 Models and Datasets

### Datasets
| Name | Description | Link |
| --- | --- | --- |
| GenRef-wds | WebDataset format of full GenRef | [HuggingFace](https://huggingface.co/datasets/diffusion-cot/GenRef-wds) |
| GenRef-CoT | Chain-of-Thought reflection dataset | [HuggingFace](https://huggingface.co/datasets/diffusion-cot/GenRef-CoT) |

### Models
| Name | Description | Finetune Data | Link |
| --- | --- | --- | --- |
| FLUX Corrector | Main FLUX-based "text image -> image" model | GenRef-wds | [HuggingFace](https://huggingface.co/diffusion-cot/FLUX-Corrector) |
| Reflection Generator | Qwen-based reflection generator | GenRef-CoT | [HuggingFace](https://huggingface.co/diffusion-cot/Reflection-Generator) |
| Image Verifier | Qwen-based image verifier | GenRef-CoT  | [HuggingFace](https://huggingface.co/diffusion-cot/Image-Verifier) |


## 🤖 Reflection Tuning

[`train_flux/config.yaml`](./train_flux/config.yaml) exposes all the arguments to control
all the training-time configurations. 

First, get the data. You can either download the `webdataset` shards from [`diffusion-cot/GenRef-wds`](https://huggingface.co/datasets/diffusion-cot/GenRef-wds) or directly pass URLs.

When using local paths, set `path` under `[train][dataset]` to a glob pattern: `DATA_DIR/genref_*.tar`. The current `config.yaml` configures training to stream from the `diffusion-cot/GenRef-wds` repository. You can even
change the number of tars you want to stream for easier debugging. Just change `genref_{0..208}.tar` to something
like `genref_{0..4}.tar`, depending on the number of shards you want to use.

Run the following command for training the FLUX Corrector:

```bash
bash train_flux/train.sh
```

We tested our implementation on a single node of 8 80GB A100s and H100s. We acknowledge that there are opportunities
for optimization, but we didn't prioritize them in this release.

>[!NOTE]
> Validation during training is yet to be implemented.

## ⚡ Inference Time Scaling

### Introduction
We provide the code for the inference time scaling of our reflection-tuned models. Currently, we support:
* GPT-4o as verifier, reflection generator, and prompt refiner.
* [NVILA-2B](https://huggingface.co/Efficient-Large-Model/NVILA-Lite-2B-Verifier) verifier from SANA.
* Our [reflection generator](https://huggingface.co/diffusion-cot/Reflection-Generator).

### Setup
First, you need to set up the following:

```bash
export OPENAI_API_KEY=your_api_key
# if you want to use NVILA as verifier
pip install transformers==4.46
pip install git+https://github.com/bfshi/scaling_on_scales.git
```
Then you need to set up the `FLUX_PATH` and `LORA_PATH` in the config file of `tts/config`. The `FLUX_PATH` is basically the contents of [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main) which can be downloaded like so:

```py
from huggingface_hub import snapshot_download

local_dir = "SOME_DIR"
snapshot_download(repo_id="black-forest-labs/FLUX.1-dev", local_dir=local_dir)
```

The `LORA_PATH` is our [corrector model](https://huggingface.co/diffusion-cot/FLUX-Corrector) path.

If you want to use our finetuned reflection generator, you need to first install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Then download
the model from [here](https://huggingface.co/diffusion-cot/Reflection-Generator) and change the `model_name_or_path` in the config file of
`tts/config/our_reflectionmodel.yaml` to the reflection generator path. To be specific, the path should be like `Reflection-Generator/infer/30000`. Next, host the model with:

```bash
API_PORT=8001 CUDA_VISIBLE_DEVICES=0 llamafactory-cli api configs/our_reflectionmodel.yaml
```
And change the `name` of `reflection_args` in the config file to `ours`.

### Run
First, please run `tts_t2i_baseline.py` to generate naive noise scaling results, with the commands:
```bash
export OUTPUT_DIR=output_dir
cd tts
python tts_t2i_noise_scaling.py --output_dir=$OUTPUT_DIR --meta_path=geneval/evaluation_metadata.jsonl --pipeline_config_path=configs/flux.1_dev_gptscore.json 
```

Next, you can run the following command to generate the results of reflection tuning:
```bash
python tts_reflectionflow.py --imgpath=$OUTPUT_DIR --pipeline_config_path=configs/flux.1_dev_nvilascore.json --output_dir=NEW_OUTPUT_DIR
```

We also provide the code for only noise & prompt scaling:
```bash
python tts_t2i_noise_prompt_scaling.py --output_dir=$OUTPUT_DIR --meta_path=geneval/evaluation_metadata.jsonl --pipeline_config_path=configs/flux.1_dev_gptscore.json 
```

### NVILA Verifier Filter

After generation, we provide the code using NVILA verifier to filter and get different numbers of sample results.
```bash
python verifier_filter.py --imgpath=$OUTPUT_DIR --pipeline_config_path=configs/flux.1_dev_nvilascore.json 
```

## 🤝 Acknowledgement

We are deeply grateful for the following GitHub repositories, as their valuable code and efforts have been incredibly helpful:

* OminiControl (https://github.com/Yuanshi9815/OminiControl)
* Flux-TTS (https://github.com/sayakpaul/tt-scale-flux)


## ✏️ Citation

If you find ReflectionFlow useful for your research and applications, please cite using this BibTeX:

```bibtex
@misc{zhuo2025reflectionperfectionscalinginferencetime,
      title={From Reflection to Perfection: Scaling Inference-Time Optimization for Text-to-Image Diffusion Models via Reflection Tuning}, 
      author={Le Zhuo and Liangbing Zhao and Sayak Paul and Yue Liao and Renrui Zhang and Yi Xin and Peng Gao and Mohamed Elhoseiny and Hongsheng Li},
      year={2025},
      eprint={2504.16080},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.16080}, 
}
```
