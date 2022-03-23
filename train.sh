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

source vqa/bin/activate
cd mmf

mmf_run config="configs/experiments/pilot/with_grid.yaml" \
model=pilot \
dataset=okvqa \
run_type=train_val
