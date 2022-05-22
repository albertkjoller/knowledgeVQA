#!/bin/sh
#BSUB -J pilot_single_mul_5000
#BSUB -o /work3/s194262/save/models/5000_steps/pilot_single_mul/output_file_%J.out
#BSUB -e /work3/s194262/save/models/5000_steps/pilot_single_mul/error_file_%J.err
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
source vqa2/bin/activate
cd mmf

mmf_run config='configs/experiments/pilot/single_mul.yaml' \
    datasets=okvqa \
    model=qlarifais \
    run_type=train_val \
    env.data_dir=/work3/s194262/torch/mmf/data \
    env.cache_dir=/work3/s194262/torch/mmf \
    env.save_dir=/work3/s194262/save/models/5000_steps/pilot_single_mul \
    training.seed=1\
    training.tensorboard=1 \
    training.evaluation_interval=5000 \
    checkpoint.max_to_keep=-1 \
    optimizer.params.lr=0.0005 \
    optimizer.params.weight_decay=1e-6 \


