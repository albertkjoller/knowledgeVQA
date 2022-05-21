
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
from mmexp.utils.tools import fetch_test_embeddings

def plot_TSNE(model, data):
    
    # Get test embeddings
    test_embeddings = fetch_test_embeddings(model)
    
    # number of categories
    num_categories = 10
    
    

    # TODO: create something more generic for stratification function...
    
    
    # start words
    start_words = data.question_tokens.apply(lambda x: x[0])
    categories = list(zip(*Counter(start_words).most_common(num_categories)))[0]
    cat2idx = {cat: i for i, cat in enumerate(categories)}
    
    # Get test data
    test_labels = start_words[start_words.apply(lambda x: x in categories)].reset_index(drop=True)
    test_data = pd.DataFrame(test_embeddings).loc[:, start_words.apply(lambda x: x in categories)].to_numpy()
    
    # Compute t-SNE
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(test_embeddings.T)
    
    # Scatter plot with colors based on the stratification_func
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8,8))
    for cat in categories:
        indices = test_labels[test_labels == cat].index.to_numpy()
        color = cmap(cat2idx[cat])
        ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(color).reshape(1,4), label = cat ,alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.show()
