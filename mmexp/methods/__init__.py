#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:06:27 2022

@author: s194253
"""

from .qlarifais.object_removal import ObjectRemoval as OR
from .torchray.multimodal_extremal_perturbation import multi_extremal_perturbation as MMEP
from .torchray.multimodal_gradient import multimodal_gradient as MMGradient
from .torchray.multimodal_gradcam import multimodal_gradcam as MMGradCAM
from .qlarifais.random_noise import random_image as VisualNoise
from .qlarifais.random_noise import random_question as TextualNoise
from .qlarifais.attention_map import attention_map

__all__ = [
    "OR",
    "VisualNoise",
    "TextualNoise",
    "MMEP",
    "MMGradient",
    "MMGradCAM",   
    "attention_map",
]