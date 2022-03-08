import torchvision
import os
import yaml
from mmf.models.first_model import First_Model
from omegaconf import OmegaConf
import torch
import collections
# from mmf.utils.logger import setup_logger, setup_very_basic_config
from torch.utils.mobile_optimizer import optimize_for_mobile

def saveModel(loadedModelName, saveModelName, config, savePath = ''):

    loadedModelPath = os.getcwd()+"/save/models/"+loadedModelName+".pth"

    model = First_Model(config)

    model.load_state_dict(torch.load(loadedModelPath))

    #model = torch.load(os.getcwd()+"/save/models/"+modelName+".pth")
    model.eval()

    # Mobilenet_v2 quantized model for phones
    model_quantized = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=True)

    # Trace the model
    torchscript_model = torch.jit.trace(model_quantized, model)

    # Optimizing it for mobile:
    torchscript_model_optimized = optimize_for_mobile(torchscript_model)

    # Saving the model
    filePath = savePath + saveModelName + ".pt"

    # Create folder if non-existant
    if savePath != '':
        os.makedirs(savePath, exist_ok=False)

    torch.jit.save(torchscript_model_optimized, filePath)


def loadConfig(modelDefaultsPath = '/Users/philliphoejbjerg/Desktop/UNI/6.semester/Bachelors_project/Github/explainableVQA/mmf/mmf/configs/models/first_model/defaults.yaml'):
    """
    Loads the config yaml file
    """
    with open(modelDefaultsPath, 'r') as stream:
        try:
            parsed_yaml=yaml.safe_load(stream)
            print(parsed_yaml)
        except yaml.YAMLError as exc:
            print(exc)

    # Extract model name
    model_name = modelDefaultsPath.split("/")[-2]

    config = OmegaConf.create(parsed_yaml['model_config'][model_name])

    return config

config = loadConfig()

config = unfoldDict(config)

config = OmegaConf.create(config)

saveModel("first_model_final", "firstModelPhone", config, "/Users/philliphoejbjerg/Desktop/UNI/6.semester/Bachelors_project/Github/explainableVQA/mmf/save/models")
