#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:29:22 2022

@author: s194253
"""

from torchray.attribution.gradient import gradient

def multimodal_gradient(model, x, category_id):

    # TODO: change this to source code
    
    # Gradient method.
    saliency = gradient(model, x, category_id)
    
    return saliency
    
