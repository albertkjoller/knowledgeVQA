#!/bin/sh
#BSUB -J baseline_mul_test
#BSUB -o /work3/s194262/tests/models/optimized/baseline_mul/test_output_file_%J.out
#BSUB -e /work3/s194262/tests/models/optimized/baseline_mul/test_error_file_%J.err
#BSUB -n 6
#BSUB -q gpua100
#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -W 24:00
#BSUB -R 'rusage[mem=5GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -B
#BSUB -N


nvidia-smi
module load cuda/11.1
source vqa2/bin/activate
cd mmf

mmf_run config=/work3/s194262/save/models/optimized/ablation1_double_ama/config.yaml \
    model=qlarifais \
    dataset=okvqa \
    run_type=test \
    env.report_dir=/work3/s194262/tests/models/optimized/ablation1_double_ama/reports \
    env.tensorboard_logdir=/work3/s194262/tests/models/optimized/tensorboard/ablation1_double_ama \
    checkpoint.resume_file=/work3/s194262/save/models/optimized/ablation1_double_ama/best.ckpt

