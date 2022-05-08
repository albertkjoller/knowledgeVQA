#!/bin/sh
#BSUB -J test_run.visual_bert.bs512.s1.adam_w.lr1e-05.mu22000.fbFalse
#BSUB -o /Users/arond.jacobsen/Documents/GitHub/explainableVQA/mmf/tools/sweeps/save/test_runvisual_bert.bs512.s1.adam_w.lr1e-05.mu22000.fbFalse.ngpu1/output_file_%J.out
#BSUB -e /Users/arond.jacobsen/Documents/GitHub/explainableVQA/mmf/tools/sweeps/save/test_runvisual_bert.bs512.s1.adam_w.lr1e-05.mu22000.fbFalse.ngpu1/error_file_%J.err
#BSUB -n 1
#BSUB -gpu num=1
#BSUB -W 05:00
#BSUB -R
#BSUB -B
#BSUB -N


nvidia-smi
module load cuda/11.1
source vqa2/bin/activate
cd mmf


python3 -u /Users/arond.jacobsen/Documents/GitHub/explainableVQA/mmf/mmf/../mmf_cli/run.py distributed.world_size 1 checkpoint.resume True env.save_dir /Users/arond.jacobsen/Documents/GitHub/explainableVQA/mmf/tools/sweeps/save/test_runvisual_bert.bs512.s1.adam_w.lr1e-05.mu22000.fbFalse.ngpu1 run_type train_val config projects/visual_bert/configs/vqa2/defaults.yaml training.num_workers 5 dataset vqa2 model visual_bert training.batch_size 512 training.seed 1 scheduler.type warmup_cosine scheduler.params.num_warmup_steps 2000 scheduler.params.num_training_steps 22000 optimizer.type adam_w optimizer.params.lr 1e-05 optimizer.params.eps 1e-08 training.max_updates 22000 training.log_format json training.pin_memory True training.log_interval 1000 training.checkpoint_interval 1000 training.evaluation_interval 4000 training.find_unused_parameters True model_config.visual_bert.freeze_base False

wait $! 
sleep 610 & 
wait $!