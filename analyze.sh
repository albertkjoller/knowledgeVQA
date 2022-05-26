#!/bin/sh
#BSUB -J analyze_ablation1_single_mul
#BSUB -o /zhome/b8/5/147299/Desktop/explainableVQA/analyze_files/temp_output_file_%J.out
#BSUB -e /zhome/b8/5/147299/Desktop/explainableVQA/analyze_files/temp_error_file_%J.err
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

python qlarifais_analyze.py --model_dir /work3/s194262/save/models/optimized/baseline_ama --torch_cache /work3/s194262 --stratify_by start_words okvqa_categories question_length answer_length numerical_answers num_visual_objects visual_objects_types --test True --report_dir /work3/s194262/tests/models/optimized/baseline_ama/reports --pickle_path /work3/s194262/tests/models/optimized/baseline_ama --save_path /work3/s194262/results/baseline_ama --okvqa_file /work3/s194262/OKVQA_rich.json



