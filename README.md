# Visual Question Answering - top-down attention and explainability

Project repository related to bachelor thesis on Technical University of Denmark (DTU) in spring 2022. Investigating the role of top-down attention mechanisms, knowledge graphs and how interpretable these are through explainability tools.

Mention OK-VQA!

<p align="center">
  <img src="https://github.com/albertkjoller/explainableVQA/blob/main/imgs/pipeline/baseline.png" alt="drawing" width="600"/>
</p>

The project is built as an instance of the [MMXAI module](https://gitlab.doc.ic.ac.uk/g207004202/explainable-multimodal-classification) that is based on the [MMF-framework](https://github.com/facebookresearch/mmf).

## Setup

###TODO: Describe that we use mmf-modular framework and mmxai modular framework

Note: this project was carried out with access to GPU cores, for which reason the setup for reproducibility will also require access to GPU.

Clone the repository in your terminal and change directory.

    git clone https://github.com/albertkjoller/explainableVQA.git
    cd explainableVQA

Create a virtual environment in your preferred way - e.g. using `conda` - and activate it.

    conda create -n vqa python=3.8
    conda activate vqa

<!-- Install dependencies from the MMXAI module (following their [installation guide](https://gitlab.doc.ic.ac.uk/g207004202/explainable-multimodal-classification)). 

    pip install --editable .
    pip install -r requirements.txt
-->

Install the MMF-module.

    cd mmf
    pip install --editable .
    cd ..

Install specific dependencies used for this project...

    pip install torch==1.8.0+cu111 torchvision torchaudio torchtext -f https://download.pytorch.org/whl/torch_stable.html

    pip install torch-scatter==2.0.8 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.8.0+cu111.html

    pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html

And that's it! Now, you should be able to reproduce the findings of this project!
	

## Pre-trained models

Download pre-trained models [here](https://drive.google.com/drive/folders/17o9YjWwAQ0rtvYC5QKM6TI_0yHcu6iSY?usp=sharing).
After downloading, create the folder `explainableVQA/mmf/save/` and place the downloaded `models`-folder here.

### Predicting with a pre-trained model

Now, you're ready to use the pre-trained model for predictions! Change directory to the `explainableVQA/examples`-folder, run the following command and follow the in-prompt directions.

    cd examples
    python Qlarifais_predict.py

## Training your own model

Training your own model is very easy! There are a few steps you need to complete before running a model:

1) create a new script (`explainableVQA/mmf/mmf/models/my_new_model.py`)
2) implement model class in this script (e.g. `MyNewModel`)
3) create default configurations (as a YAML-file, place it in `explainableVQA/mmf/mmf/configs/models/my_new_model/defaults.yaml`)

Of course, exploiting the complete functionality of the MMF-framework requires digging deeper into the folders, experiment-configurations, etc.. However, you should now be able to run the trian command from the `explainableVQA/mmf`-folder!

    cd mmf

    mmf_run config='configs/experiments/model_name/run_type.yaml' \
        datasets=okvqa \
        model={my_new_model} \
        run_type=train_val \
        env.data_dir={where_you_want_to_store_the_data}/torch/mmf/data \
        env.save_dir={where_you_want_to_save_the_model}/save/models/{and_the_save_name} \
        trainer.params.gpus=1 \

# fast run (Phillip) example (REMOVE LATER!):
mmf_run config='configs/experiments/baseline/mul.yaml' \
    datasets=okvqa \
    model=qlarifais \
    run_type=train_val \
    env.data_dir=/work3/s194253/torch/mmf/data \
    env.cache_dir=/work3/s194253/torch/mmf \
    env.save_dir=/work3/s194253/save/models/fast_training \
    training.max_updates=1 \
    training.max_epochs=None \
    trainer.params.gpus=1 \
