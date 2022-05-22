#!/bin/sh
#BSUB -J pilot_double_ama_final
#BSUB -o /work3/s194253/save/models/pilot_double_ama/output_file_%J.out
#BSUB -e /work3/s194253/save/models/pilot_double_ama/error_file_%J.err
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

python3 -u /zhome/b8/5/147299/Desktop/explainableVQA/mmf/mmf/../mmf_cli/run.py \
    env.save_dir /work3/s194253/models/pilot_double_ama \ 
    env.cache_dir /work3/s194253/torch/mmf \
    env.data_dir /work3/s194253/torch/mmf/data \ 
    env.tensorboard_logdir /work3/s194253/save/models/pilot_double_ama  \
    training.tensorboard 1 \
    run_type train_val \
    config /zhome/b8/5/147299/Desktop/explainableVQA/mmf/mmf/configs/experiments/pilot/double_ama.yaml \
    model qlarifais \
    dataset okvqa \
    training.seed 1 \
    optimizer.params.lr 0.0005 \
    optimizer.params.weight_decay 1e-06 \
    model_config.qlarifais.classifier.params.dropout 0.3 \ 
    model_config.qlarifais.fusion.params.dropout 0.1 \

