#!/bin/bash

# 参数配置
total_start=0     # 总起始索引
total_end=560      # 总结束索引
step=26           # 每个作业处理的数据量

# 循环提交作业
for ((current_start=$total_start; current_start<$total_end; current_start+=$step)); do
  current_end=$((current_start + $step))
  
  # 处理最后一个区间
  if [ $current_end -gt $total_end ]; then
    current_end=$total_end
  fi

  # 动态生成作业名和输出文件
  job_name="verifier_${current_start}to${current_end}"
  output_file="verifier_${current_start}to${current_end}.%J.out"

  # 用heredoc提交作业
  sbatch <<EOF
#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J $job_name
#SBATCH -o $output_file
#SBATCH --mail-user=liangbing.zhao@kaust.edu.sa  
#SBATCH --mail-type=ALL
#SBATCH --time=10:30:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=a100
#SBATCH --exclude=gpu201-16-r,gpu201-09-l,gpu101-09-l,gpu203-09-l
#SBATCH --account conf-neurips-2025.05.22-elhosemh

# source ~/anaconda3/bin/activate llama_factory
# API_PORT=8001 CUDA_VISIBLE_DEVICES=0 llamafactory-cli api /home/zhaol0c/uni_editing/ReflectionFlow/LLama_Factory/inference.yaml &

module load cuda/12.1
source ~/anaconda3/bin/activate vllm_a100
# export OPENAI_API_KEY=sk-hlepj5ZdRYki30qQKYPLuxrr_Na6t2co28omDnG4D7T3BlbkFJziPP9Z6Id_xewMotG7z6ZeclofyBcRJkwuuKSCZo0A
export OPENAI_API_KEY=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
# export OPENAI_API_KEY=sk-RwCsrhDRGprjKgzlXNhUShhRukDsCAxkrrapbBpfHY2dYlyw

# python tts_ti2i_neworder_sequential_ourverifier_ourreflectionmodel.py \\
#   --output_dir=/ibex/user/zhaol0c/uniediting_continue/nvilaverifier_exps/b2_d16_6000model_ourverifier \\
#   --start_index=$current_start \\
#   --end_index=$current_end \\
#   --imgpath=/ibex/user/zhaol0c/uniediting/training/geneval_gtsamples/ \\
#   --pipeline_config_path configs/flux.1_dev_modified.json

# python tts_t2i_refine_sequential.py \\
#   --output_dir=/ibex/user/zhaol0c/uniediting_continue/nvilaverifier_exps/ttsrefine_nfe32 \\
#   --start_index=$current_start \\
#   --end_index=$current_end \\
#   --meta_path=/home/zhaol0c/uni_editing/geneval/prompts/evaluation_metadata.jsonl \\
#   --pipeline_config_path configs/flux.1_dev_modified.json

# python tts_t2i_baseline.py \\
#   --output_dir=/ibex/user/zhaol0c/uniediting_continue/nvilaverifier_exps/baseline_nfe32 \\
#   --start_index=$current_start \\
#   --end_index=$current_end \\
#   --meta_path=/home/zhaol0c/uni_editing/geneval/prompts/evaluation_metadata.jsonl \\
#   --pipeline_config_path configs/flux.1_dev_modified.json

# python verifier_filter.py --imgpath /ibex/user/zhaol0c/uniediting_continue/nvilaverifier_exps/b2_d16_6000model_ourverifier --start_index $current_start --end_index $current_end --pipeline_config_path configs/flux.1_dev_modified.json 

python verifier_filter_forsana.py --imgpath /ibex/user/zhaol0c/uniediting_continue/sana_exps/Sana/output/geneval_generated_path --start_index $current_start --end_index $current_end --pipeline_config_path configs/flux.1_dev_modified.json 
EOF

  echo "已提交作业: $job_name (索引范围: ${current_start}-${current_end})"
done