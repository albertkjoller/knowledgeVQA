#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 18:30:53 2022

@author: s194253
"""

import numpy as np
import pandas as pd
import torch

from mmf.models import Qlarifais

from mmexp.analyzer import prediction_dataframe, plot_TSNE, stratified_predictions
from mmexp.utils.tools import paths_to_okvqa

# obtain user input
save_dir = input("Enter directory path of saved models ('save'-folder): ")
model_name = input("\nEnter saved model filename: ")
path_to_torch_cache = input("\nEnter the path to where your torch-cache folder is located (e.g. /work3/s194253): ")

okvqa_filepath = input("\nEnter filepath to OK-VQA data file including filename. If None is passed, default test-data will be used: ")

model = Qlarifais.from_pretrained(f"{save_dir}/models/{model_name}", path_to_torch_cache)
model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

# load okvqa
data_path, images_path = paths_to_okvqa(model, run_type='test')
if okvqa_filepath == 'None':
    okvqa_filepath = data_path
okvqa_test = pd.DataFrame.from_records(np.load(okvqa_filepath, allow_pickle=True)[1:])


# Plot t-SNE
plot_TSNE(model, data=okvqa_test, label_by='start_word')

# Get predictions
data = prediction_dataframe(model, data=okvqa_test)

# Stratifications



