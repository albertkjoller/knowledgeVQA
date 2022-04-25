#!/bin/sh
#BSUB -J answer_conceptnet
#BSUB -o answer_conceptnet_%J.out
#BSUB -e answer_conceptnet_%J.err
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=32G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -B
#BSUB -N

source vqa/bin/activate

python data_investigation/extract_conceptnet.py
