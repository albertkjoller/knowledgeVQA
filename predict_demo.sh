#!/bin/sh
#BSUB -J predict_demo
#BSUB -o predict_demo.out
#BSUB -e predict_demo.err
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=32G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -B
#BSUB -N

source vqa/bin/activate

cd utils
export PYTHONPATH=.
cd ..

python mmf/predict_demo.py
