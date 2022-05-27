#!/bin/sh
#BSUB -J baseline_ama_test
#BSUB -o /work3/s194253/results/baseline_ama/test_output_file_%J.out
#BSUB -e /work3/s194253/results/baseline_ama/test_error_file_%J.err
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

cd ..
cd examples

python qlarifais_run_analyze.py --model_dir /work3/s194262/save/models/optimized/baseline_ama \
    --torch_cache /work3/s194253 \
    --report_dir /work3/s194253/results/baseline_ama/reports \
    --save_path /work3/s194253/results/baseline_ama \
    --okvqa_file /work3/s194253/OKVQA_rich.json
    --stratify_by start_words okvqa_categories question_length answer_length numerical_answers num_visual_objects visual_objects_types \

python qlarifais_run_explainer.py --model_dir /work3/s194262/save/models/optimized/baseline_ama \
    --torch_cache /work3/s194253 \
    --report_dir /work3/s194253/results/baseline_ama/reports \
    --save_path /work3/s194253/results/baseline_ama \
    --protocol_dir /work3/s194262/protocol \
    --analysis_type OR VisualNoise TextualNoise \
    --explainability_methods MMGradient \
    --protocol_name pilotQ.txt \
    --show_all True \

