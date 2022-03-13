import os
import collections
from pathlib import Path
import pickle

import torch, torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

from utils.config import loadConfig
from mmf.models.first_model import First_Model

from mmf.utils.general import get_current_device

from mmf.common.sample import Sample, SampleList

def saveModel(loadedModelName, saveModelName):

    config = loadConfig(loadedModelName)#

    model = First_Model(config)
    model.build()
    model.init_losses()

    loadedModelPath = Path(f"{os.path.dirname(os.getcwd())}/mmf/save/models/{('_').join(loadedModelName.split('_')[:-1])}/{loadedModelName}.pth")
    model.load_state_dict(torch.load(loadedModelPath))

    model.to(get_current_device())

    #model = torch.load(os.getcwd()+"/save/models/"+modelName+".pth")
    model.eval()

    """
    # Mobilenet_v2 quantized model for phones
    model_quantized = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=True)
    """

    # Trace the model
    example_input = preparePickle()
    torchscript_model = torch.jit.script(model)

#    torchscript_model = torch.jit.trace(model, example_input) # or torch.jit.script(model)?

    # evt. ændre sample_list["dataset_name"], sample_list["dataset_type"], manuelt i linje 123 i losses.py?

    # Optimizing it for mobile:
    torchscript_model_optimized = optimize_for_mobile(torchscript_model)

    # Saving the model

    savePath = f"{os.getcwd()}/phoneModels/{saveModelName}.pt"

    torch.jit.save(torchscript_model_optimized, savePath)

def preparePickle(
        pickle_path='/Users/philliphoejbjerg/Desktop/UNI/6.semester/Bachelors_project/Github/explainableVQA/mmf/Sample_first_model2.pkl'):
    file = open(pickle_path, 'rb')
    Sample_list = pickle.load(file)

    sample = Sample()

    for key, value in Sample_list.items():
        if isinstance(value[0], str):
            pass # sample[key] = value
        elif isinstance(value[0], list):
            pass
        else:
            sample[key] = value[0]

    example_input = SampleList([sample]).to(get_current_device())

    return example_input

example_input = preparePickle()





saveModel("first_model_final", "firstModelPhone")

