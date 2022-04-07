#!/bin/sh
#BSUB -J pilot
#BSUB -o pilot_%J.out
#BSUB -e pilot_%J.err
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=32G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -B
#BSUB -N

source vqa_cuda/bin/activate
cd mmf

module load cuda/10.2
/appl/cuda/10.2/samples/NVIDIA_CUDA-10.2_Samples/bin/x86_64/linux/release/deviceQuery

mmf_run config='configs/experiments/pilot/test.yaml' \
    datasets=okvqa \
    model=pilot \
    run_type=train_val \
    env.data_dir=/work3/s194262/torch/mmf/data \
    env.save_dir=/work3/s194262/save/models/pilot_grids \
    trainer.params.gpus=1 \