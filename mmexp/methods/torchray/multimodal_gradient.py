#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:29:22 2022

@author: s194253
"""

from .attribution.common import gradient_to_saliency, resize_saliency


def multimodal_gradient(model, image_object,
                        question, category_id, 
                        context_builder=None, 
                        resize=False, resize_mode='bilinear',
                        **kwargs):
    
    # Gradient method.
    y = model.classify(image_object, question, explain=True)
    z = y[0, category_id]
    z.backward()
    
    image_tensor = model.image_tensor
    
    saliency = gradient_to_saliency(image_tensor)
    saliency = resize_saliency(image_tensor, 
                               saliency, 
                               resize, 
                               mode=resize_mode,
                               )
    
    return saliency




    """Gradient method

    The function takes the same arguments as :func:`.common.saliency`, with
    the defaults required to apply the gradient method, and supports the
    same arguments and return values.
    """
    
    """
    assert context_builder is None
    return multimodal_saliency(model,
                               image_tensor,
                               image_path,
                               question,
                               category_id,
                               resize=True,
                               **kwargs,
                               )
    """
    
