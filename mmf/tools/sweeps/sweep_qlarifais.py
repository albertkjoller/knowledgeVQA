#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

import lib as sweep
from lib import hyperparam

'''

python tools/sweeps/sweep_qlarifais.py \
--backend lsf \
--resume_finished \
--resume_failed \
--checkpoints_dir /Users/arond.jacobsen/Desktop/test \
-t -1 \
-g 8 \
-n 2 \
--constraint some_constraint \
--comment "test run" \
-q my_job \



python tools/sweeps/sweep_visual_bert.py \
--resume_finished \
--resume_failed \
--checkpoints_dir /work3/s194262/save \
-t 1 \
-g 8 \
-n 2 \
--constraint some_constraint \
--comment "test run" \
--partition some_partition \
-p my_job \
--backend slurm \

'''




def get_grid(args):
    # For list of args, run `python tools/sweeps/{script_name}.py --help`.

    return [
        hyperparam("run_type", "train_val"),
        hyperparam(
            "training.batch_size", [512, 256], save_dir_key=lambda val: f"bs{val}"
        ),
        hyperparam("training.seed", 1, save_dir_key=lambda val: f"s{val}"),
        hyperparam(
            "optimizer.params.lr", [5e-5, 1e-5], save_dir_key=lambda val: f"lr{val}"
        ),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
