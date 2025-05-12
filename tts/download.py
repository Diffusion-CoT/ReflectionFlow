from huggingface_hub import snapshot_download

local_dir = "/mnt/data-volume/FLUX.1-dev"
snapshot_download(repo_id="black-forest-labs/FLUX.1-dev", local_dir=local_dir)

local_dir = "/mnt/data-volume/FLUX-Corrector"
snapshot_download(repo_id="diffusion-cot/FLUX-Corrector", local_dir=local_dir)

local_dir = "/mnt/data-volume/Reflection-Generator"
snapshot_download(repo_id="diffusion-cot/Reflection-Generator", local_dir=local_dir)
