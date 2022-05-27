#!/bin/sh
#BSUB -J explainer_baseline_ama
#BSUB -o /zhome/b8/5/147299/Desktop/explainableVQA/analysis_files/temp_output_file_%J.out
#BSUB -e /zhome/b8/5/147299/Desktop/explainableVQA/analysis_files/temp_error_file_%J.err
#BSUB -n 6
#BSUB -q gpua100
#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -W 24:00
#BSUB -R 'rusage[mem=4GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -B
#BSUB -N

nvidia-smi
module load cuda/11.1
source vqa2/bin/activate
cd examples

python qlarifais_run_explainer.py --model_dir /work3/s194262/save/models/optimized/baseline_ama \
    --torch_cache /work3/s194262 \
    --report_dir /work3/s194262/results/baseline_ama/reports \
    --save_path /work3/s194262/results/baseline_ama \
    --protocol_dir /work3/s194262/protocol \
    --analysis_type OR VisualNoise TextualNoise \
    --explainability_methods MMGradient \
    --protocol_name pilotQ.txt \
    --show_all True \



