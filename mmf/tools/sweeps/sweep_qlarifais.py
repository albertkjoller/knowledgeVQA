#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

import lib as sweep
from lib import hyperparam

'''

python mmf/tools/sweeps/sweep_visual_bert.py \
--baseline_model mmf/models/qlarifais.py \
--backend lsf \
--resume_failed \
--checkpoints_dir /work3/s194262/save/sweeps \
--run_type train_val \
--config configs/experiments/baseline/mul.yaml \
-prefix testrun \
-t -1 \
-n 1 \
-q gpua100 \
-gpus "num=1:mode=exclusive_process" \
-R "rusage[mem=128G]" \
-W 05:00 \


'''




def get_grid(args):
    # For list of args, run `python tools/sweeps/{script_name}.py --help`.

    # initialize, contain all params
    hp = []

    # input commands
    hp.extend([hyperparam("run_type", args.run_type), hyperparam("config", args.config)])

    # general hyperparams
    hp.extend([hyperparam("optimizer.params.lr", [0.001, 0.0001], save_dir_key=lambda val: f"lr{val}")])

    # experiment specific hyperparams
    if args.config.split('/')[-2] == 'baseline':
        hp.extend([hyperparam('model_config.qlarifais.fusion.params.dropout', [0.1, 0.2, 0.3],
                              save_dir_key=lambda val: f"lr{val}")])
    # todo: seed if test

    return hp



def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
