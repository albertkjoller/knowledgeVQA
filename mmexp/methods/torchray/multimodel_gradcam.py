#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:25:34 2022

@author: s194253
"""


def image2tensor(image_path):
    # convert image to torch tensor with shape (1 * 3 * 224 * 224)
    img = Image.open(image_path)
    p = transforms.Compose([transforms.Scale((224, 224))])

    img, i = imsc(p(img), quiet=False)
    return torch.reshape(img, (1, 3, 224, 224))
