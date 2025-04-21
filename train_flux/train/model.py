import os
import lightning as L
from diffusers.pipelines import FluxPipeline
import torch
from torchvision import transforms
from peft import LoraConfig, get_peft_model_state_dict

import prodigyopt

from flux.generate import generate
from flux.transformer import tranformer_forward
from flux.condition import Condition
from flux.pipeline_tools import encode_images, prepare_text_input


class OminiModel(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        data_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        save_path: str = None,
        run_name: str = None,
        cache_dir: str = None,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.data_config = data_config
        self.save_path = save_path
        self.run_name = run_name

        # Load the Flux pipeline
        self.flux_pipe: FluxPipeline = (
            FluxPipeline.from_pretrained(flux_pipe_id, cache_dir=cache_dir, torch_dtype=dtype).to(device)
        )
        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # Freeze the Flux pipeline
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()

        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path, lora_config)

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # # TODO: Implement this
            # raise NotImplementedError
            self.flux_pipe.load_lora_weights(lora_path, adapter_name="default")
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = []
            for name, p in self.transformer.named_parameters():
                if "lora" in name:
                    lora_layers.append(p)
            # lora_layers = filter(
            #     lambda p: p.requires_grad, self.transformer.parameters()
            # )
        else:
            if lora_config.get("target_modules", None) == "all-linear":
                target_modules = set()
                for name, module in self.transformer.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        target_modules.add(name)
                target_modules = list(target_modules)
                lora_config["target_modules"] = target_modules
            self.transformer.add_adapter(LoraConfig(**lora_config))
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers) if not isinstance(lora_layers, list) else lora_layers

    def save_lora(self, path: str):
        FluxPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def validation_step(self, batch, batch_idx):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(42)
        target_size = self.data_config["target_size"]
        condition_size = self.data_config["condition_size"]
        prompt = batch["description"][0]
        original_prompt = batch["original_prompt"][0]
        condition_type = batch["condition_type"][0]
        condition_img = batch["condition"][0]
        to_pil = transforms.ToPILImage()
        condition_img = to_pil(condition_img.cpu().float())
        position_delta = batch["position_delta"][0]
        condition = Condition(
            condition_type=condition_type,
            condition=condition_img,
            position_delta=position_delta,
        )
        
        res = generate(
            self.flux_pipe,
            prompt=original_prompt,
            prompt_2=prompt,
            conditions=[condition],
            height=target_size,
            width=target_size,
            generator=generator,
            model_config=self.model_config,
            default_lora=True,
        )
        os.makedirs(os.path.join(self.save_path, self.run_name, "val"), exist_ok=True)
        res.images[0].save(
            os.path.join(self.save_path, self.run_name, "val", f"{self.global_step}_{condition_type}_{batch_idx}.jpg")
        )

    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def step(self, batch):
        imgs = batch["image"]
        conditions = batch["condition"]
        condition_types = batch["condition_type"]
        prompts = batch["original_prompt"]
        position_delta = batch["position_delta"][0]
        prompts_2 = batch["description"]

        # Prepare inputs
        with torch.no_grad():
            # Prepare image input
            x_0, img_ids = encode_images(self.flux_pipe, imgs)

            # Prepare text input
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_pipe, prompts, prompts_2=prompts_2
            )
            
            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)
            
            # Prepare conditions
            condition_latents, condition_ids = encode_images(self.flux_pipe, conditions)

            # Add position delta
            condition_ids[:, 1] += position_delta[0]
            condition_ids[:, 2] += position_delta[1]

            # Prepare condition type
            condition_type_ids = torch.tensor(
                [
                    Condition.get_type_id(condition_type)
                    for condition_type in condition_types
                ]
            ).to(self.device)
            condition_type_ids = (
                torch.ones_like(condition_ids[:, 0]) * condition_type_ids[0]
            ).unsqueeze(1)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        # Forward pass
        transformer_out = tranformer_forward(
            self.transformer,
            # Model config
            model_config=self.model_config,
            # Inputs of the condition (new feature)
            condition_latents=condition_latents,
            condition_ids=condition_ids,
            condition_type_ids=condition_type_ids,
            # Inputs to the original transformer
            hidden_states=x_t,
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]

        # Compute loss
        loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()
        return loss
