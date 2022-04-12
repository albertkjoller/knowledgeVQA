import os
from pathlib import Path
import torch, torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from model_utils.config import loadConfig
from mmf.models.pilot import Pilot
from mmf.models.first_model import First_Model
import numpy as np

import pickle

def saveModel(modelFolderName, saveModelName, modelClass):

    # Path of loaded model
    loadedModelPath = Path(f"{os.getcwd()}/save/models/{('_').join(modelFolderName.split('_'))}/{('_').join(modelFolderName.split('_')[:-1])}_final.pth")

    # Load config file
    # str(modelClass).lower().split(".")[-1].split("'")[0] --> "pilot"
    config = loadConfig(modelFolderName, str(modelClass).lower().split(".")[-1].split("'")[0])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Building model
    model = modelClass(config)
    model.build()
    model.init_losses()
    model.load_state_dict(torch.load(loadedModelPath, map_location=torch.device('cpu')), strict = False)
    model.to(device)
    model.eval()

    # Trace the model
    #torchscript_model = torch.jit.script(model)

    #with open('/Users/philliphoejbjerg/Desktop/UNI/6.semester/Bachelors_project/Github/explainableVQA/mmf/filename.pickle', 'rb') as handle:
    #    sample_list = pickle.load(handle)

    #sample_list['text_len'] = torch.from_numpy(np.array(sample_list['text_len']))

    #torchscript_model = torch.jit.trace(model, example_inputs = sample_list, strict = False)
    torchscript_model = torch.jit.script(model)

    # Optimizing it for mobile:
    torchscript_model_optimized = optimize_for_mobile(torchscript_model)

    # Saving the model
    savePath = f"{os.getcwd()}/phoneModels/{saveModelName}.pt"
    torch.jit.save(torchscript_model_optimized, savePath)

    print(f'Success: {saveModelName} has been saved at {savePath}')

saveModel("pilot_grids", "pilot_test_phone", Pilot)

