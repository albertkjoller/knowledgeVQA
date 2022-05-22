

## Testing the best model
Running the best found hyper parameters for an experiment on the test set.


Assuming:
- A hyperparameter search is done.
- The best hyperparameters have been estimated maunally from Tensorboards.

Then follow this simple guide:
1) Retrieve the command given in the README of the main repository for testing and predicting. 
    1.1 If running through LSF update open the `test.sh`- and `predict.sh`-files.
2) Specify the model checkpoint path and modify paths for your needs.
3) Run the command from the terminal.
    3.1 If LSF-user: first submit the test-file by `bsub < test.sh`. When job is done, submit the prediction-file by `bsub < predict.sh`.

## Training the optimal model  

If you want more checkpoints than the `best.ckpt`, do the following:

1) Retrieve the command given in the README of the main repository for training.
    1.1 If running through LSF update open the `train.sh`-file.
2) Specify the `evaluation_interval`-parameter and set the `checkpoint.max_to_keep` to the desired amount of checkpoints for saving.
3) Change the model name in `env.save_dir` to not overwrite already trained models.
3) Run the command from the terminal.
    3.1 If LSF-user: submit the train-file by `bsub < train.sh`.

## Steps on updating to the best module 

<!---
### After Baseline
In `experiments/modules/fusions/...`:
- Set the optimal dropout value in the fusion modules.
    - In: `two_modality_airthmetic.yaml` and `two_modality_ama.yaml`.
--->

### Before Ablation 1
<!---
In `experiments/modules/fusions/...`:
- Set the optimal dropout value in the fusion modules.
   - In: `triple_modality_arithmetic.yaml`, `double_two_modality_airthmetic.yaml` and `double_two_modality_ama.yaml`.
    
In `experiments/modules/classifiers/...`:
- Set the optimal dropout value in the classifier module.
    - In: `embeddings.yaml`.
--->

In `experiments/...`:
- Import the best fusion module for upcoming experiments.
   - In: `ablation1/...`, `ablation2/...` and `ablation3/...`.

### After Ablation 1
In `experiments/...`:
- Import the best image encoder in the upcoming experiments.
   - Experiments: `ablation2/...` and `ablation3/...`.

### After Ablation 2
In `experiments/ablation2/...`:
- Set the optimal dropout value in the attention fusion modules in:
   - `graph_ama.yaml`, `graph_mul.yaml`, `question_ama.yaml` and `question_mul.yaml`.

### After Ablation 3
In `experiments/ablation3/...`:
- Set the optimal dropout value in the attention fusion modules in:
   - `q_g_double_ama.yaml`, `q_g_double_mul.yaml`, `q_g_single_mul.yaml`

---

## Debugging failed/stopped runs

### If one or more runs need to restart

1) Delete the errorneous folders in `save/sweeps/...` of experiments that need to restart.
   
    1.1) if the run had previously started, delete its' corresponding tensorboard folder too.
   
2) Run the full hyperparameter sweep command again - valid folders will be skipped, i.e. not overwriting finished/running experiments.

### If you're out of memory

Bad luck! Solve it manually by removing large files and consider checkpointing less regularly or delete non-optimal models that were saved in the hyperparameter sweep.

<!---
### If a run suddenly stops and should continue from `current.ckpt`
These steps are only intended for a single job:
1) Copy its submit file commands in `train.log` which are after the `running commands:` and paste it in the `sweep_var.sh`.
2) Adjust the `.out` and `.err` file names.
3) Add the `checkpoint.resume True` argument to the `mmf_run` type command 
4) Submit the job to clusters.
--->



