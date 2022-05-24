#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 17:03:35 2022

@author: s194253
"""

import os
from pathlib import Path

import numpy as np

from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt


def plot_TSNE(stratified_object, model_name: str, save_path: str):
        
    # Compute t-SNE
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(stratified_object.embeddings.T)
    
    # Scatter plot with colors based on the stratification_func
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8,8))
    for cat in stratified_object.categories:
        indices = stratified_object.data['stratification_label'][stratified_object.data['stratification_label'] == cat].index.to_numpy()
        color = cmap(stratified_object.cat2idx[cat])
        ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(color).reshape(1,4), label = cat ,alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    
    # Set title and legend
    plt.axis('off')
    plt.title(f"Qlarifais-model: {model_name}\
              \nPrediction embeddings stratified by {stratified_object.by}",
              fontsize=12,
              loc='left')
    plt.legend(loc='center', bbox_to_anchor=(0.95, 0.3, 0.5, 0.5))
    plt.tight_layout()

    # Save fig
    save_path = Path(save_path)
    os.makedirs(save_path / 'tsne', exist_ok=True)
    plt.savefig(save_path / f'tsne/{model_name}_{stratified_object.by}.png')
    
    # Show fig
    #plt.show()


def embedding_variation(model, modularity: str):
    
    if modularity == 'graph':
        pass
    elif modularity == 'question':
        pass
    elif modularity == 'image':
        pass
    
    
    # delta_e = 
    
    # delta_f = 
    
    # delta_e / delta_f
    
    raise NotImplementedError