
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

import cv2
import torch

from mmf.models import Qlarifais
import torchvision.datasets.folder as tv_helpers
from mmexp.utils.tools import fetch_test_embeddings


# obtain user input
save_dir = input("Enter directory path of saved models ('save'-folder): ")
model_name = input("\nEnter saved model filename: ")
path_to_torch_cache = input("\nEnter the path to where your torch-cache folder is located (e.g. /work3/s194253): ")

model = Qlarifais.from_pretrained(f"{save_dir}/models/{model_name}", path_to_torch_cache)
model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

# Get test embeddings
test_embeddings = fetch_test_embeddings(model)

from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt

tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(test_embeddings.T)


# paths to data
data_path = Path(model.config.dataset_config.okvqa.data_dir) / 'okvqa'
test_data_path = data_path / 'defaults/annotations/annotations/imdb_test.npy'
images_path = data_path / 'defaults/images'

# test dataset
okvqa_test = pd.DataFrame.from_records(np.load(test_data_path, allow_pickle=True)[1:])


# Plot those points as a scatter plot and label them based on the pred labels
cmap = cm.get_cmap('tab20')
fig, ax = plt.subplots(figsize=(8,8))

num_categories = 10
for lab in range(num_categories):
    indices = test_predictions==lab
    ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
ax.legend(fontsize='large', markerscale=2)
plt.show()


print("")