conda init
source ~/.bashrc
conda activate /ceph/data-bk/miniconda3/envs/flux

cd /ceph/data-bk/zl/ReflectionFlow/train_flux

export TOKENIZERS_PARALLELISM=true
accelerate launch --main_process_port 41353 -m train.train
