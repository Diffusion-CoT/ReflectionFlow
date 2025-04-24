from torch.utils.data import DataLoader
import torch
import lightning as L
import yaml
import os
import time

from diffusers.utils.logging import set_verbosity_error
set_verbosity_error()

from .data_wds import ImageConditionWebDataset
from .model import OminiModel
from .callbacks import TrainingCallback

def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank

def get_config():
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=wandb_config["name"] if wandb_config["name"] else run_name,
            config={},
            settings=wandb.Settings(start_method="fork"),
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


def main():
    # Initialize
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    config = get_config()
    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")

    # Initialize WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)

    # Initialize dataset and dataloader
    if training_config["dataset"]["type"] == "img":
        dataset = ImageConditionWebDataset(
            training_config["dataset"]["path"],
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            condition_type=training_config["condition_type"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
            drop_image_prob=training_config["dataset"]["drop_image_prob"],
            drop_reflection_prob=training_config["dataset"]["drop_reflection_prob"],
            root_dir=training_config["dataset"]["root_dir"],
            split_ratios=training_config["dataset"]["split_ratios"],
            training_stages=training_config["dataset"]["training_stages"],
        )
        if "val_path" in training_config["dataset"]:
            val_dataset = ImageConditionWebDataset(
                training_config["dataset"]["val_path"],
                condition_size=training_config["dataset"]["condition_size"],
                target_size=training_config["dataset"]["target_size"],
                condition_type=training_config["condition_type"],   
                root_dir=training_config["dataset"]["val_root_dir"],
                drop_text_prob=0,
                drop_image_prob=0,
                drop_reflection_prob=0,
                shuffle_buffer=0,
            )
        else:
            val_dataset = None
    else:
        raise NotImplementedError

    train_loader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        num_workers=training_config["dataloader_workers"],
    )
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
    else:
        val_loader = None

    # Try add resume training
    lora_path = None

    if training_config['resume_training_from_last_checkpoint'] and os.path.exists(training_config['save_path']):
        # get latest directory in training_config['save_path'], ignore hidden files
        all_training_sessions = [d for d in os.listdir(training_config['save_path']) if not d.startswith('.')]
        all_training_sessions.sort(reverse=True)
        last_training_session = all_training_sessions[0]
        if os.path.exists(f"{training_config['save_path']}/{last_training_session}/ckpt"):
            ckpt_paths = [d for d in os.listdir(f"{training_config['save_path']}/{last_training_session}/ckpt") if not d.startswith('.')]
            ckpt_paths.sort(reverse=True)
            lora_path = f"{training_config['save_path']}/{last_training_session}/ckpt/{ckpt_paths[0]}"
            print(f"Resuming training from {lora_path}")
        else:
            print("No checkpoint found. Training without LoRA weights.")

    elif training_config['resume_training_from_checkpoint_path'] != "":
        _lora_path = training_config['resume_training_from_checkpoint_path']
        # Check if the path exists
        if os.path.exists(_lora_path):
            lora_path = _lora_path
            print(f"Training with LoRA weights from {_lora_path}")
        else:
            print(f"Path {_lora_path} does not exist. Training without LoRA weights.")
    
    # Initialize model
    trainable_model = OminiModel(
        flux_pipe_id=config["model_path"],
        lora_path=lora_path,
        lora_config=training_config["lora_config"],
        data_config=training_config["dataset"],
        device=f"cuda:{rank}",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        save_path=training_config.get("save_path", "./output"),
        run_name=run_name,
        cache_dir=config["cache_dir"],
    )

    # Callbacks for logging and saving checkpoints
    training_callbacks = (
        [TrainingCallback(run_name, training_config=training_config)]
        if is_main_process
        else []
    )

    # Initialize trainer
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=training_callbacks,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
        val_check_interval=training_config.get("sample_interval", 1000),
        num_sanity_val_steps=training_config.get("num_sanity_val_steps", -1),
    )

    setattr(trainer, "training_config", training_config)

    # Save config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}")
        os.makedirs(f"{save_path}/{run_name}/val")
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Start training
    trainer.fit(trainable_model, train_loader, val_loader)


if __name__ == "__main__":
    main()
