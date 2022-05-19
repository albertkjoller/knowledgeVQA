

## Testing the best model
Running the best found hyper parameters for an experiment on the test set.


Assuming:
- A hyper parameter search is done.
- The best hyper parameters have been found using the tensorboards.
- The best model checkpoint has been chosen.
    - If the `best.ckpt` update value seems arbitrary, consider the other eligible checkpoints.

Then:
1) Go to the best models' folder and its `train.log`file.
2) Copy the information given by `running_commands:` (i.e. the .sh commands).
3) Paste this in the `sweep_var.sh` file.
4) Adjusting the `python3 -u` command:
   
    4.1 Modify the following: `run_type train_val` to `run_type test`
    
    4.2 Add the following: `resume_file best.ckpt` or e.g. `resume_file models/model_900.ckpt`

6) submit this folder to the cluster: `bsub<sweep_var.sh`

---

## Steps on updating to the best module 
Setting the optimal hyper parameter for each module, and importing the best found modules in upcoming experiments.

### After Baseline
In `experiments/modules/fusions/...`:
- Set the optimal dropout value in the fusion modules.
   - In: `two_modality_airthmetic.yaml` and `two_modality_ama.yaml`.

### After Pilot
In `experiments/modules/fusions/...`:
- Set the optimal dropout value in the fusion modules.
   - In: `triple_modality_arithmetic.yaml`, `double_two_modality_airthmetic.yaml` and `double_two_modality_ama.yaml`.
    
In `experiments/modules/classifiers/...`:
- Set the optimal dropout value in the classifier module.
    - In: `embeddings.yaml`.

In `experiments/...`:
- Import the best triple modality fusion module in upcoming experiments.
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



