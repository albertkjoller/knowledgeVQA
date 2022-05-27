#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:03:04 2022

@author: s194253
"""

from PIL import Image
import numpy as np

import nltk
nltk.download('brown')
from nltk.corpus import brown


def random_image(input_image):
    
    input_image = np.array(input_image)
    x1 = input_image.shape[1]
    x2 = input_image.shape[0]
    
    gaussian_R = np.random.normal(0, 1, (x1, x2)) * 255
    gaussian_G = np.random.normal(0, 1, (x1, x2)) * 255
    gaussian_B = np.random.normal(0, 1, (x1, x2)) * 255
    gaussian = np.array([gaussian_R, gaussian_G, gaussian_B]).T

    PIL_image = Image.fromarray(np.uint8(gaussian)[::-1]).convert('RGB')

    return PIL_image

def random_question(orig_question, model):
    random_words = np.random.choice(brown.words(), len(orig_question.split(" ")))
    return (" ").join(random_words)