
import os
from pathlib import Path

import torch
import cv2

from mmf.models import Qlarifais



# obtain user input
save_dir = input("Enter directory path of saved models ('save'-folder): ")
model_name = input("\nEnter saved model filename: ")
path_to_torch_cache = input("\nEnter the path to where your torch-cache folder is located (e.g. /work3/s194253): ")

model = Qlarifais.from_pretrained(f"{save_dir}/models/{model_name}", path_to_torch_cache)
model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


import numpy as np
import pandas as pd

data_path = Path(model.config.dataset_config.okvqa.data_dir) / 'okvqa'
test_data_path = data_path / 'defaults/annotations/annotations/imdb_test.npy'
images_path = data_path / 'defaults/images'

okvqa_test = pd.DataFrame.from_records(np.load(test_data_path, allow_pickle=True)[1:])


image = (images_path / okvqa_test.image_name[0]).as_posix() + '.jpg'

img = cv2.imread(image)

#outputs = model.classify(image=image, text=question)

print("")