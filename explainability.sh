#!/bin/sh
#BSUB -J 0explainability_
#BSUB -o 0explainability_%J.out
#BSUB -e 0explainability_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=128G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 00:10
#BSUB -B
#BSUB -N

nvidia-smi
module load cuda/11.1

source /work3/s194253/envs/vqa/bin/activate
cd mmf

mmf_run config='configs/experiments/baseline/mul.yaml' \
    datasets=okvqa \
    model=qlarifais \
    run_type=train_val \
    env.data_dir=/work3/s194253/torch/mmf/data \
    env.cache_dir=/work3/s194253/torch/mmf \
    env.save_dir=/work3/s194253/save/models/temporary \
    training.max_updates=1 \
    training.max_epochs=None \
    trainer.params.gpus=1 \
