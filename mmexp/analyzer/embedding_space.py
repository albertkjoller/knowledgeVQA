#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 17:03:35 2022

@author: s194253
"""

import os
import numpy as np

from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt


def plot_TSNE(stratified_object, save_path: str):
        
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
    plt.title(stratified_object.by)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 0., 0.5, 0.5))

    # Save fig
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(save_path)

    # Show fig
    plt.show()


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