#!/bin/sh
#BSUB -J baseline_ama_mmf_test
#BSUB -o /work3/s194253/results/baseline_ama/mmf_log.out
#BSUB -e /work3/s194253/results/baseline_ama/mmf_log.err
#BSUB -n 6
#BSUB -q gpuv100
#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -W 2:00
#BSUB -R 'rusage[mem=5GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -B
#BSUB -N


nvidia-smi
module load cuda/11.1
source /work3/s194253/envs/vqa/bin/activate
cd mmf

mmf_run config=/work3/s194262/save/models/optimized/baseline_ama/config.yaml \
    model=qlarifais \
    dataset=okvqa \
    run_type=test \
    env.report_dir=/work3/s194253/results/baseline_ama/reports \
    env.tensorboard_logdir=/work3/s194253/results/tensorboard/baseline_ama \
    env.save_dir=/work3/s194253/results/baseline_ama/mmf \
    checkpoint.resume_file=/work3/s194262/save/models/optimized/baseline_ama/best.ckpt

