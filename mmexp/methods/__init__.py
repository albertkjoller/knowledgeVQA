#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:06:27 2022

@author: s194253
"""

from .torchray.multimodal_extremal_perturbation import multi_extremal_perturbation as MMEP
from .torchray.multimodal_gradient import multimodal_gradient as MMGradient
from .qlarifais.object_removal import ObjectRemoval as MMGradientOR
from .torchray.multimodal_gradcam import multimodal_gradcam as MMGradCAM

__all__ = [
    "MMEP",
    "MMGradient",
    "MMGradientOR",
    "MMGradCAM",
    
]