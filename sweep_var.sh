#!/bin/sh
#BSUB -J testrun.qlarifais.lr0.0001.lr0.2
#BSUB -o /Users/arond.jacobsen/Documents/GitHub/explainableVQA/mmf/save/testrunqlarifais.lr0.0001.lr0.2.ngpu1/output_file_%J.out
#BSUB -e /Users/arond.jacobsen/Documents/GitHub/explainableVQA/mmf/save/testrunqlarifais.lr0.0001.lr0.2.ngpu1/error_file_%J.err
#BSUB -n 1
#BSUB -q gpua100
#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -W 05:00
#BSUB -R 'rusage[mem=128G]'
#BSUB -B
#BSUB -N


nvidia-smi
module load cuda/11.1
source vqa2/bin/activate
cd mmf


python3 -u /Users/arond.jacobsen/Documents/GitHub/explainableVQA/mmf/mmf/../mmf_cli/run.py distributed.world_size 1 checkpoint.resume True env.save_dir /Users/arond.jacobsen/Documents/GitHub/explainableVQA/mmf/save/testrunqlarifais.lr0.0001.lr0.2.ngpu1 run_type train_val config /Users/arond.jacobsen/Documents/GitHub/explainableVQA/mmf/mmf/configs/experiments/baseline/mul.yaml model qlarifais dataset okvqa optimizer.params.lr 0.0001 model_config.qlarifais.fusion.params.dropout 0.2


wait $! 
sleep 610 & 
wait $!