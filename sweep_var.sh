#!/bin/sh
#BSUB -J ama.train_val.seed1.lr5e-05.wd1e-06.cdo0.3.fdo0.1
#BSUB -o /work3/s194253/save/sweeps/ama/train_val.seed1.lr5e-05.wd1e-06.cdo0.3.fdo0.1/output_file_%J.out
#BSUB -e /work3/s194253/save/sweeps/ama/train_val.seed1.lr5e-05.wd1e-06.cdo0.3.fdo0.1/error_file_%J.err
#BSUB -n 6
#BSUB -q gpuv100
#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -W 24:00
#BSUB -R 'rusage[mem=4GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -B
#BSUB -N


nvidia-smi
module load cuda/11.1
source /work3/s194253/envs/vqa/bin/activate
cd mmf


python3 -u /zhome/b8/5/147299/Desktop/explainableVQA/mmf/mmf/../mmf_cli/run.py env.save_dir /work3/s194253/save/sweeps/ama/train_val.seed1.lr5e-05.wd1e-06.cdo0.3.fdo0.1 env.cache_dir /work3/s194253/torch/mmf env.data_dir /work3/s194253/torch/mmf/data run_type train_val config /zhome/b8/5/147299/Desktop/explainableVQA/mmf/mmf/configs/experiments/baseline/ama.yaml model qlarifais dataset okvqa training.seed 1 optimizer.params.lr 5e-05 optimizer.params.weight_decay 1e-06 model_config.qlarifais.classifier.params.dropout 0.3 model_config.qlarifais.fusion.params.dropout 0.1


wait $! 
sleep 610 & 
wait $!