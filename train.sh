#!/bin/sh
#BSUB -J predictive_vqa_demo
#BSUB -o predictive_vqa_demo_%J.out
#BSUB -e predictive_vqa_demo_%J.err
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=32G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -B
#BSUB -N

source vqa/bin/activate
cd mmf

mmf_run config="configs/experiments/predictive_vqa/defaults.yaml" \
model=predictive_vqa \
dataset=okvqa \
run_type=train_val
