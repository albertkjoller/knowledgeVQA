#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

import lib as sweep
from lib import hyperparam

# todo: adjust config, prefix, run_type
'''
python mmf/tools/sweeps/sweep_qlarifais.py \
--run_type train_val \
--resume False \
--config /zhome/96/8/147177/Desktop/explainableVQA/mmf/mmf/configs/experiments/extra/losses/bce_contrastive.yaml \
-prefix bce_contrastive \
--baseline_model /zhome/96/8/147177/Desktop/explainableVQA/mmf/mmf/models/qlarifais.py \
--backend lsf \
--checkpoints_dir /work3/s194262/save/sweeps \
--cache_dir /work3/s194262/torch/mmf \
--data_dir /work3/s194262/torch/mmf/data \
-t -1 \
-n 6 \
-q gpua100 \
-gpus "num=1:mode=exclusive_process" \
-R "rusage[mem=5GB]" \
-W 24:00 \
'''




def get_grid(args):
    # For list of args, run `python3 tools/sweeps/{script_name}.py --help`.

    # initialize, contain all params
    hp = []
    # input commands and set up
    hp.extend([hyperparam("run_type", args.run_type, save_dir_key=lambda val: val), hyperparam("config", args.config),
               hyperparam("model", "qlarifais"), hyperparam("dataset", "okvqa"),
               hyperparam("training.seed", 1, save_dir_key=lambda val: f"seed{val}"),
               ])


    # --- parameters to optimize ---

    # general hyperparams
    # learning rate (lr)
    hp.extend([hyperparam("optimizer.params.lr", [5e-3, 5e-4], save_dir_key=lambda val: f"lr{val}")])
    # weight decay (wd) 0.0001
    hp.extend([hyperparam("optimizer.params.weight_decay", [1e-4, 1e-6], save_dir_key=lambda val: f"wd{val}")])

    # classifier hp search
    # hidden dimension (chd)
    #hp.extend([hyperparam("model_config.qlarifais.classifier.params.h_dim", [2500, 5000], save_dir_key=lambda val: f"chd{val}")])
    # number of non-linear layers (cnl)
    #hp.extend([hyperparam("model_config.qlarifais.classifier.params.num_non_linear_layers", [2, 4], save_dir_key=lambda val: f"cnl{val}")])

    # experiment specific hp search
    experiment_type = args.config.split('/')[-2] # extracting experiment folder name
    # hp search for optimal fusion module
    if experiment_type in ['baseline', 'pilot', 'ablation1']:
        # these experiements vary in input
        # dropout (fdo)
        hp.extend([hyperparam('model_config.qlarifais.fusion.params.dropout', [0.1, 0.3],
                              save_dir_key=lambda val: f"fdo{val}")])

        # dropout (cdo)
        hp.extend([hyperparam("model_config.qlarifais.classifier.params.dropout", [0.1, 0.3], save_dir_key=lambda val: f"cdo{val}")])

        # fusion hidden dimension (fhd)
        #hp.extend([hyperparam('model_config.qlarifais.fusion.params.h_dim', [2500, 5000],
        #           save_dir_key=lambda val: f"fhd{val}")])

    # optimal fusion module has been chosen, now sweep it in attention
    if experiment_type in ['ablation2', 'ablation3']:
        # these experiements do not vary in input, thus use previous optimized fusion modules
        # dropout (ado)
        hp.extend([hyperparam('model_config.qlarifais.attention.params.fusion.params.dropout', [0.1, 0.3],
                              save_dir_key=lambda val: f"ado{val}")])

        # fusion hidden dimension (ahd)
        #hp.extend([hyperparam('model_config.qlarifais.attention.params.fusion.params.h_dim', [2500, 5000],
        #           save_dir_key=lambda val: f"ahd{val}")])

    # sweeping lambda values
    if experiment_type == 'losses':
        # dropout (ado)
        hp.extend([hyperparam('model_config.qlarifais.losses[0].params.lambda_bce', [1, 1/300, 1/2250],
                              save_dir_key=lambda val: f"lbce{val}")])


    return hp



def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
