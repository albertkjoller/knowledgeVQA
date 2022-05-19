
import os
from pathlib import Path

import numpy as np
import pandas as pd

import cv2
import torch

from mmf.models import Qlarifais
import torchvision.datasets.folder as tv_helpers



# obtain user input
save_dir = input("Enter directory path of saved models ('save'-folder): ")
model_name = input("\nEnter saved model filename: ")
path_to_torch_cache = input("\nEnter the path to where your torch-cache folder is located (e.g. /work3/s194253): ")

model = Qlarifais.from_pretrained(f"{save_dir}/models/{model_name}", path_to_torch_cache)
model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


# paths to data
data_path = Path(model.config.dataset_config.okvqa.data_dir) / 'okvqa'
test_data_path = data_path / 'defaults/annotations/annotations/imdb_test.npy'
images_path = data_path / 'defaults/images'

# test dataset
okvqa_test = pd.DataFrame.from_records(np.load(test_data_path, allow_pickle=True)[1:])

for i in range(okvqa_test.__len__()):
    # load test image
    img_name = (images_path / okvqa_test.image_name[i]).as_posix() + '.jpg'
    image = tv_helpers.default_loader(img_name)
    
    # Get test question
    question = okvqa_test.question_str[i]
    
    # Get predicted embedding
    outputs = model.classify(image=image, text=question, embedding_output=True)

print("")