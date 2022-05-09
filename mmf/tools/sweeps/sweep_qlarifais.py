#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

import lib as sweep
from lib import hyperparam

# todo: adjust config, prefix, run_type
'''
python mmf/tools/sweeps/sweep_qlarifais.py \
--run_type train_val \
--config /zhome/96/8/147177/Desktop/explainableVQA/mmf/mmf/configs/experiments/baseline/ama.yaml \
-prefix ama \
--baseline_model /zhome/96/8/147177/Desktop/explainableVQA/mmf/mmf/models/qlarifais.py \
--backend lsf \
--resume_failed \
--checkpoints_dir /work3/s194262/save/sweeps \
--cache_dir /work3/s194262/torch/mmf \
--data_dir /work3/s194262/torch/mmf/data \
-t -1 \
-n 4 \
-q gpua100 \
-gpus "num=1:mode=exclusive_process" \
-R "rusage[mem=128G]" \
-W 05:00 \
'''

'''
# on mac
python mmf/tools/sweeps/sweep_qlarifais.py \
--baseline_model /Users/arond.jacobsen/Documents/GitHub/explainableVQA/mmf/mmf/models/qlarifais.py \
--backend lsf \
--resume_failed \
--checkpoints_dir /Users/arond.jacobsen/Documents/GitHub/explainableVQA/mmf/save \
--cache_dir
--data_dir
--run_type train_val \
--config /Users/arond.jacobsen/Documents/GitHub/explainableVQA/mmf/mmf/configs/experiments/baseline/mul.yaml \
-prefix testrun \
-t -1 \
-n 1 \
-q gpua100 \
-gpus "num=1:mode=exclusive_process" \
-R "rusage[mem=128G]" \
-W 05:00 \
'''




def get_grid(args):
    # For list of args, run `python3 tools/sweeps/{script_name}.py --help`.

    # initialize, contain all params
    hp = []
    # input commands and set up
    hp.extend([hyperparam("run_type", args.run_type, save_dir_key=lambda val: val), hyperparam("config", args.config),
               hyperparam("model", "qlarifais"), hyperparam("dataset", "okvqa"),
               hyperparam("training.seed", 1, save_dir_key=lambda val: f"s{val}"),
               ])


    # --- parameters to optimize ---

    # general hyperparams
    # learning rate (lr)
    hp.extend([hyperparam("optimizer.params.lr", [0.0001, 0.001], save_dir_key=lambda val: f"lr{val}")])
    # weight decay (wd)
    #hp.extend([hyperparam("optimizer.params.weight_decay", [1e-5, 1e-7, 1e-8, 1e-10], save_dir_key=lambda val: f"wd{val}")])
    # todo: scheduler?

    # we keep batch size, epochs(only look at best validation)...??

    # experiment specific hyperparams
    if args.config.split('/')[-2] == 'baseline':
        # fusion dropout (fdo)
        hp.extend([hyperparam('model_config.qlarifais.fusion.params.dropout', [0.1, 0.2],
                              save_dir_key=lambda val: f"fdo{val}")])
        # fusion hidden dim (fhd)
        #hp.extend([hyperparam('model_config.qlarifais.fusion.params.h_dim', [2048, 5000],
        #           save_dir_key=lambda val: f"fhd{val}")])


    if args.config.split('/')[-2] == 'pilot':
        # fusion dropout (fdo)
        hp.extend([hyperparam('model_config.qlarifais.fusion.params.dropout', [0.1, 0.2],
                              save_dir_key=lambda val: f"fdo{val}")])
        # fusion hidden dim (fhd)
        #hp.extend([hyperparam('model_config.qlarifais.fusion.params.h_dim', [2048, 5000],
        #           save_dir_key=lambda val: f"fhd{val}")])

    # todo add more for each experiment

    return hp



def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
