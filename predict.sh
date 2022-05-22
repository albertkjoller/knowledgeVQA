#!/bin/sh
#BSUB -J pilot_double_ama_predict
#BSUB -o /work3/s194253/tests/models/pilot_double_ama/predict_output_file_%J.out
#BSUB -e /work3/s194253/tests/models/pilot_double_ama/predict_error_file_%J.err
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

mmf_predict config=/work3/s194253/save/models/pilot_double_ama/config.yaml \
    model=qlarifais \
    dataset=okvqa \
    run_type=test \
    env.report_dir=/work3/s194253/tests/models/pilot_double_ama/reports \
    env.tensorboard_logdir=/work3/s194253/tests/models/pilot_double_ama \
    checkpoint.resume_file=/work3/s194253/save/models/pilot_double_ama/best.ckpt

