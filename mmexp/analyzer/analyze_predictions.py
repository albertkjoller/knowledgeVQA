#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 18:10:09 2022

@author: s194253
"""

import pandas as pd
from pathlib import Path

from mmexp.utils.tools import load_predictions


def prediction_dataframe(model, data):
    # Get predictions
    predictions = load_predictions(model)
    predictions = predictions.rename(columns={'answer': 'prediction'})
    
    # Merge with test data
    data = data.merge(predictions, on='question_id')
    data = data[['question_id', 'question_tokens', 
                 'answers', 'prediction', 'topk',
                 'image_name']]
    return data

def stratified_predictions(model, data, stratification_func=None):
    
    predictions = load_predictions(model)
    
    
    
    return predictions

