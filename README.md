<div align="center" style="font-family: charter;">
<h1><i>From Reflection to Perfection:</i>:</br>Scaling Inference-Time Optimization for Text-to-Image Diffusion Models via Reflection Tuning</h1>



<a href="tmp" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-ReflectionFlow-red?logo=arxiv" height="20" /></a>
<a href="https://liangbingzhao.github.io/reflection2perfection/" target="_blank">
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

- [2025/4/??] Release [paper](tmp).
- [2025/4/??] Release GenRef dataset, as well as the training and evaluation code.

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

## 🚀 Model Zoo and Dataset
| Dataset | Link |
| --- | --- |
| GenRef-CoT | [GenRef-CoT](https://huggingface.co/datasets/diffusion-cot/GenRef-CoT) |
| GenRef-wds | [GenRef-wds](https://huggingface.co/datasets/diffusion-cot/GenRef-wds) |

| Model | Link |
| --- | --- |
| ReflectionFlow | [ReflectionFlow](https://huggingface.co/diffusion-cot/experimental-models) |
| Our Reflection Generation Model | [Our Reflection Generation Model](https://huggingface.co/diffusion-cot/reflection-models) |

### Introduction

Coming soon.

### Evaluation on VLM

Coming soon.

### Evaluation on LMM

Coming soon.


## 🤖 Reflection Tuning

The config file is `config.yaml`. Run the following command for training:
```bash
bash train/script/train_subject.sh
```

## ⚡ Inference Time Scaling

### Introduction
We provide the code for the inference time scaling of our reflection-tuned models. Currently, we support:
* OpenAI as score verifier, reflection generator, and prompt refiner.
* NVILA as score verifier.
* Our finetuned reflection generator.

### Setup
First, you need to set up with following command lines:
```bash
export OPENAI_API_KEY=your_api_key
# if you want to use NVILA as score verifier
pip install transformers==4.46
pip install git+https://github.com/bfshi/scaling_on_scales.git
```
Then you need to set up the FLUX_PATH and LORA_PATH in the config file of `tts/config`.

If you want to use our finetuned reflection generator, you need to first install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Then download the model from [here](https://huggingface.co/diffusion-cot/reward-models/tree/main) and change the `model_name_or_path` in the config file of `tts/config/our_reflectionmodel.yaml`. Next, host the model with
```bash
API_PORT=8001 CUDA_VISIBLE_DEVICES=0 llamafactory-cli api configs/our_reflectionmodel.yaml
```
And change the `name` of `reflection_args` in the config file to `ours`.

### Run
First, please run `tts_t2i_baseline.py` to generate naive noise scaling results, with the commands:
```bash
cd tts
python tts_t2i_baseline.py --output_dir=OUTPUT_DIR --meta_path=geneval/evaluation_metadata.jsonl --pipeline_config_path=configs/flux.1_dev_gptscore.json 
```

Next, you can run the following command to generate the results of reflection tuning:
```bash
python tts_t2i_noise_scaling.py --imgpath=OUTPUT_DIR --pipeline_config_path=CONFIG_PATH --output_dir=NEW_OUTPUT_DIR
```

We also privide the code for only noise&prompt scaling:
```bash
python tts_t2i_noise_prompt_scaling.py --output_dir=OUTPUT_DIR --meta_path=geneval/evaluation_metadata.jsonl --pipeline_config_path=configs/flux.1_dev_gptscore.json 
```

### Nvila Verifier Filter

After generation, we provide the code using nvila verifier to filter getting different number of samples results.
```bash
python verifier_filter.py --imgpath=OUTPUT_DIR --pipeline_config_path=configs/flux.1_dev_nvilascore.json 
```

## 🤝 Acknowledgement

We are deeply grateful for the following GitHub repositories, as their valuable code and efforts have been incredibly helpful:

* OminiControl (https://github.com/Yuanshi9815/OminiControl)
* Flux-TTS (https://github.com/sayakpaul/tt-scale-flux)


## ✏️ Citation

If you find ReflectionFlow useful for your your research and applications, please cite using this BibTeX:

```bibtex
tmp
```
