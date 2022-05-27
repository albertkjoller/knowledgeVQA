#!/bin/sh
#BSUB -J ablation3_q_g_double_mul.train_val.seed1.lr0.0005.wd1e-06.ado0.3
#BSUB -o /work3/s194253/save/sweeps/ablation3_q_g_double_mul/train_val.seed1.lr0.0005.wd1e-06.ado0.3/output_file_%J.out
#BSUB -e /work3/s194253/save/sweeps/ablation3_q_g_double_mul/train_val.seed1.lr0.0005.wd1e-06.ado0.3/error_file_%J.err
#BSUB -n 6
#BSUB -q gpua100
#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -W 24:00
#BSUB -R 'rusage[mem=5GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -B
#BSUB -N


nvidia-smi
module load cuda/11.1
source /work3/s194253/envs/vqa/bin/activate
cd mmf


python3 -u /zhome/b8/5/147299/Desktop/explainableVQA/mmf/mmf/../mmf_cli/run.py env.save_dir /work3/s194253/save/sweeps/ablation3_q_g_double_mul/train_val.seed1.lr0.0005.wd1e-06.ado0.3 env.cache_dir /work3/s194253/torch/mmf env.data_dir /work3/s194253/torch/mmf/data checkpoint.max_to_keep 1 training.tensorboard 1 env.tensorboard_logdir /work3/s194253/save/sweeps/tensorboard/ablation3_q_g_double_mul/train_val.seed1.lr0.0005.wd1e-06.ado0.3 run_type train_val config /zhome/b8/5/147299/Desktop/explainableVQA/mmf/mmf/configs/experiments/ablation3/q_g_double_mul.yaml model qlarifais dataset okvqa training.seed 1 optimizer.params.lr 0.0005 optimizer.params.weight_decay 1e-06 model_config.qlarifais.attention.params.fusion.params.dropout 0.3


wait $! 
sleep 610 & 
wait $!