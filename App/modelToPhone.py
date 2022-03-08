import os
import collections
from pathlib import Path

import torch, torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

from utils.config import loadConfig
from mmf.models.first_model import First_Model

def saveModel(loadedModelName, saveModelName):

    config = loadConfig(loadedModelName)#

    model = First_Model(config)

    loadedModelPath = Path(f"{os.path.dirname(os.getcwd())}/mmf/save/models/{('_').join(loadedModelName.split('_')[:-1])}/{loadedModelName}.pth")
    model.load_state_dict(torch.load(loadedModelPath))

    #model = torch.load(os.getcwd()+"/save/models/"+modelName+".pth")
    model.eval()

    # Mobilenet_v2 quantized model for phones
    model_quantized = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=True)

    # Trace the model
    torchscript_model = torch.jit.trace(model)

    # Optimizing it for mobile:
    torchscript_model_optimized = optimize_for_mobile(torchscript_model)

    # Saving the model
    filePath = savePath + saveModelName + ".pt"

    # Create folder if non-existant
    if savePath != '':
        os.makedirs(savePath, exist_ok=False)

    torch.jit.save(torchscript_model_optimized, filePath)


saveModel("first_model_final", "firstModelPhone")
