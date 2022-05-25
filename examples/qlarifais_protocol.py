#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 20:16:46 2022

@author: s194253
"""

import sys
import numpy as np
import pandas as pd
import torch

from mmf.models import Qlarifais

sys.path.append("..")
from mmexp.analyzer import *
from mmexp.utils.tools import paths_to_okvqa, str_to_class

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Script for running explainableVQA-analyses.')
    parser.add_argument(
        "--model_dir",
        required=True,
        help="path to the directory with desired model name",
    )    
    parser.add_argument(
        "--torch_cache",
        required=True,
        help="path to your torch cache directory, where the dataset is stored",
    )   
    parser.add_argument(
        "--report_dir",
        help="directory path to where the model-predictions are stored. \
            Requres --test being specified.",
        default=None,
    )
    parser.add_argument(
        "--pickle_path",
        required=True,
        help="where to temporarily store files outputted through this analysis run. \
            Currently also used for storing predicted embeddings",
        default=None,
    )
    parser.add_argument(
        "--save_path",
        required=True,
        help="where to store output of the  analysis run.",
        default=None,
    )
    parser.add_argument(
        "--explain",
        help="whether explainability tools are to be used or not. If true, \
            defaults to visualizing the gradient on the image",
        default=True,
    )
    parser.add_argument(
        "--explainability_methods",
        nargs='+',
        help="list of explainability methods to be used, i.e. ['Gradient', 'OR-Gradient']. \
            Requires --explain flag.",
        default='Gradient',
    )    
    return parser.parse_args()

if __name__ == '__main__':
    
    args = get_args()

    for exp_method in args.explainability_method:
        method = str_to_class(exp_method)