#!/bin/sh
#BSUB -J baseline_ama_512
#BSUB -o /work3/s194253/tests/baseline_ama_512/output_file_%J.out
#BSUB -e /work3/s194253/tests/baseline_ama_512/error_file_%J.err
#BSUB -n 6
#BSUB -q gpuv100
#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -W 24:00
#BSUB -R 'rusage[mem=5GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -B
#BSUB -N


nvidia-smi
module load cuda/11.1
source /work3/s194253/envs/vqa/bin/activate
cd mmf

mmf_run config=/work3/s194253/save/models/baseline_ama_512/config.yaml \
    model=qlarifais \
    dataset=okvqa \
    run_type=test \
    env.report_dir=/work3/s194253/tests/models/baseline_ama_512/reports \
    checkpoint.resume_file=/work3/s194253/save/models/baseline_ama_512/best.ckpt

