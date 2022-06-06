#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:35:53 2022

@author: s194253
"""

import numpy as np
from PIL import Image
import skimage
import matplotlib.pyplot as plt

def attention_map(image, attention, opacity=False):
    
    
    att = attention.detach().numpy()
    grid_shape = (7,7)
    im = np.transpose(image.squeeze(0).detach().numpy(), axes=[1,2,0])
    
    softmax = np.reshape(att, grid_shape)
    att_map = skimage.transform.resize(softmax, im.shape[:2], order=3)
    
    if opacity == True:
        att_map = att_map[..., np.newaxis]
        att_map = att_map*0.95+0.05
    else:
        att_map = skimage.filters.gaussian(att_map, 0.02*max(im.shape[:2]))
        att_map -= att_map.min()
        att_map /= att_map.max()
        
        # color attention map
        cmap = plt.get_cmap('jet')
        att_map = cmap(att_map)
        att_map = np.delete(att_map, 3, 2)
        
    # convert to bgr
    im = im[:, :, [2, 1, 0]]
    orig_im = (im - im.min()) / (im.max() - im.min())
    
    # plot figure
    fig, ax = plt.subplots(1, 2, dpi=400)
    ax[0].imshow(orig_im)
    ax[0].axis('off')
    ax[0].set_title('Original')
    #ax[1].imshow(im / 255)
    #ax[1].axis('off')
    #ax[1].set_title('Model input')
    ax[1].imshow(np.clip(im, 0, 255) / 255)
    ax[1].imshow(att_map, alpha=0.5)
    ax[1].axis('off')
    ax[1].set_title('Attention map')
    plt.show()