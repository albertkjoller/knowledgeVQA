# Qlarifais - Investigating Multi-Modal learning and top-down attention through explainability

*Project repository related to bachelor thesis on Technical University of Denmark (DTU) in spring 2022.*

<p align="center">
  <img src="https://github.com/albertkjoller/explainableVQA/blob/main/qlarifais_yellow.png" alt="drawing"/>
</p>

Examining multi-modal learning through experiments with external knowledge and top-down attention mechanisms by including explainability tools for the visual modality. The project aims at competing with other contributors to the [OK-VQA challenge](https://okvqa.allenai.org/leaderboard.html) by training multi-modal VQA models using the [MMF](https://github.com/facebookresearch/mmf) framework. 

## Setup

***Note***: this project was carried out with access to GPU cores which is critical for training the models. Testing model performance on the test set and through demos (`examples/qlarifais_predict.py` and `examples/qlarifais_explain.py`) can be carried out on CPU.

Clone the repository in your terminal and change directory.

    git clone https://github.com/albertkjoller/explainableVQA.git
    cd explainableVQA

Create a virtual environment in your preferred way - e.g. using `conda` - and activate it.

    conda create -n vqa python=3.8
    conda activate vqa

### MMF

Install the MMF-module.

    cd mmf
    pip install --editable .
    cd ..

Install specific dependencies used for this project that are needed for the MMF module...

#### GPU (Linux only)
    pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 torchtext==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

    pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
    
    pip install omegaconf==2.1.*

#### CPU (MacOS or Linux)
    pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 torchtext==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

    pip install “git+https://github.com/facebookresearch/detectron2.git@ffff8ac”
    
    pip install omegaconf==2.1.*

### MMEXP - Multi-Modal Explainability

So now you've got the basics for training and testing multi-modal models through MMF, however, for usingthe explainability- and analysis-tools created in this project, run the following commands:

    cd mmexp

    pip install -r requirements.txt

And that's it! Now, you should be able to reproduce the findings of this project!	

## Pre-trained models

Download pre-trained models [here](https://drive.google.com/drive/folders/17o9YjWwAQ0rtvYC5QKM6TI_0yHcu6iSY?usp=sharing).
After downloading, create the folder `explainableVQA/mmf/save/` and place the downloaded `models`-folder here.

### Predicting with a pre-trained model

Now, you're ready to use the pre-trained model for predictions! Change directory to the `explainableVQA/examples`-folder, run the following command and follow the in-prompt directions.

    cd examples
    python qlarifais_predict.py

## Training your own model

Training your own model is very easy! There are a few steps you need to complete before running a model:

1) create a new script (`explainableVQA/mmf/mmf/models/my_new_model.py`)
2) implement model class in this script (e.g. `MyNewModel`)
3) create default configurations (as a YAML-file, place it in `explainableVQA/mmf/mmf/configs/models/my_new_model/defaults.yaml`)

Of course, exploiting the complete functionality of the MMF-framework requires digging deeper into the folders, experiment-configurations, etc.. However, you should now be able to run the trian command from the `explainableVQA/mmf`-folder!
    
    cd mmf

    mmf_run config='configs/experiments/{my_new_model}/defaults.yaml' \
        datasets=okvqa \
        model={my_new_model} \
        run_type=train_val \
        env.data_dir={where_you_want_to_store_the_data}/torch/mmf/data \
        env.save_dir={where_you_want_to_save_the_model}/save/models/{and_the_desired_name} \
        trainer.params.gpus=1 \

## Testing a model

Testing a model by getting predictions on the test data from OK-VQA is as easy as training the model! Just follow this code example given by MMF.

    cd mmf

    mmf_predict config={path_to_pretrained_model}}/config.yaml \
        model={my_new_model} \
        dataset=okvqa \
        run_type=test \
        env.report_dir={where_you_want_the_results}{my_new_model}/reports \
        checkpoint.resume_file={path_to_pretrained_model}/best.ckpt
