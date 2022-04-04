# Visual Question Answering - top-down attention and explainability

Project repository related to bachelor thesis on Technical University of Denmark (DTU) in spring 2022. Investigating the role of top-down attention mechanisms, knowledge graphs and how interpretable these are through explainability tools.

<p align="center">
  <img src="https://github.com/albertkjoller/explainableVQA/blob/main/imgs/pipeline/baseline.png" alt="drawing" width="600"/>
</p>

## Pre-trained models

Download pre-trained models [here](https://drive.google.com/drive/folders/17o9YjWwAQ0rtvYC5QKM6TI_0yHcu6iSY?usp=sharing).
After downloading, create the folder `explainableVQA/mmf/save/` and place the downloaded `models`-folder here.

## Running commands
General:

    mmf_run config='configs/experiments/model_name/run_type.yaml' \
        datasets=dataset_name \
        model=model_name \
        run_type=train_val \
        env.save_dir=./save/models/your_choice

Example: 

    mmf_run config='configs/experiments/pilot/grids.yaml' \
        datasets=okvqa \
        model=pilot \
        run_type=train_val \
        env.save_dir=./save/models/airplane


For fast run you can append:

    training.max_updates=1 \
