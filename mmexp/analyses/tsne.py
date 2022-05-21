
import os
from pathlib import Path
from tqdm import tqdm

from collections import Counter
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt

import cv2
import torch

from mmf.models import Qlarifais
import torchvision.datasets.folder as tv_helpers
from mmexp.utils.tools import fetch_test_embeddings, paths_to_okvqa


# obtain user input
save_dir = input("Enter directory path of saved models ('save'-folder): ")
model_name = input("\nEnter saved model filename: ")
path_to_torch_cache = input("\nEnter the path to where your torch-cache folder is located (e.g. /work3/s194253): ")

model = Qlarifais.from_pretrained(f"{save_dir}/models/{model_name}", path_to_torch_cache)
model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

# Get test embeddings
test_embeddings = fetch_test_embeddings(model)

tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(test_embeddings.T)

# load okvqa
data_path, images_path = paths_to_okvqa(model, run_type='test')
okvqa_test = pd.DataFrame.from_records(np.load(data_path, allow_pickle=True)[1:])

# number of categories
num_categories = 10

# start words
start_words = okvqa_test.question_tokens.apply(lambda x: x[0])
categories = list(zip(*Counter(start_words).most_common(num_categories)))[0]
cat2idx = {cat: i for i, cat in enumerate(categories)}

# Get test data
test_labels = start_words[start_words.apply(lambda x: x in categories)]
test_data = pd.DataFrame(test_embeddings).loc[:, start_words.apply(lambda x: x in categories)].to_numpy()

# Plot those points as a scatter plot and label them based on the pred labels
cmap = cm.get_cmap('tab20')
fig, ax = plt.subplots(figsize=(8,8))
for lab in categories:
    indices = test_labels==lab
    ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
ax.legend(fontsize='large', markerscale=2)
plt.show()


print("")