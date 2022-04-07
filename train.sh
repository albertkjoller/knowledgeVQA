#!/bin/sh
#BSUB -J pilot_
#BSUB -o pilot__%J.out
#BSUB -e pilot__%J.err
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
/appl/cuda/11.1/samples/NVIDIA_CUDA-11.1_Samples/bin/x86_64/linux/release/deviceQuery

source vqa2/bin/activate
cd mmf

mmf_run config='configs/experiments/pilot/grids.yaml' \
    datasets=okvqa \
    model=pilot \
    run_type=train_val \
    env.data_dir=/work3/s194262/torch/mmf/data \
    env.save_dir=/work3/s194262/save/models/pilot_grids2 \
    trainer.params.gpus=1 \