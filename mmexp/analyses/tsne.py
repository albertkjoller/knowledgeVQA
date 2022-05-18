
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



outputs = model.classify(image=image, text=question, top_k=topk)