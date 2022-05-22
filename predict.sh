#!/bin/sh
#BSUB -J baseline_mul_predict
#BSUB -o /work3/s194262/tests/models/optimized/baseline_mul/predict_output_file_%J.out
#BSUB -e /work3/s194262/tests/models/optimized/baseline_mul/predict_error_file_%J.err
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
source vqa2/bin/activate
cd mmf

mmf_predict config=/work3/s194262/save/models/optimized/baseline_mul/config.yaml \
    model=qlarifais \
    dataset=okvqa \
    run_type=test \
    env.report_dir=/work3/s194262/tests/models/optimized/baseline_mul/reports \
    checkpoint.resume_file=/work3/s194262/save/models/optimized/baseline_mul/best.ckpt

