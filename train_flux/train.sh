#!/bin/bash

# Specify the config file path
export XFL_CONFIG=config.yaml

# Specify the WANDB API key
export WANDB_API_KEY="6259cb5e7b3f9ef85de258325ba564e92827f2c5"

export TOKENIZERS_PARALLELISM=true
accelerate launch --main_process_port 41353 -m train.train