#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 18:30:53 2022

@author: s194253
"""

import sys
import numpy as np
import pandas as pd
import torch

from mmf.models import Qlarifais

sys.path.append("..")
from mmexp.analyzer import prediction_dataframe, plot_TSNE, stratified_predictions, performance_report
from mmexp.utils.tools import paths_to_okvqa

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
        "--okvqa_file",
        help="path + filename of the okvqa-datafile, if not using the default \
        MMF one - e.g. if stratification on image objects is desired ",
        default=None,
    )
        
    parser.add_argument(
        "--stratify_by",
        help="how to stratifiy the data. If not specified, all test-data is used. \
            Options: ['okvqa_categories', 'start_words']",
        default=None,
    )
     
    parser.add_argument(
        "--tsne",
        help="Creates a t-SNE plot from the predicted Qlarifais-embeddings. \
            Requires stratification flag for coloring.",
        default=True,
    )
        
    parser.add_argument(
        "--performance_report",
        help="Prints a performance report on the data.",
        default=True,
    )
    
    return parser.parse_args()

if __name__ == '__main__':
    
    # Get arguments
    args = get_args()
    
    # Load model
    model = Qlarifais.from_pretrained(args.model_dir, args.torch_cache)
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    
    # Load data
    data_path, images_path = paths_to_okvqa(model, run_type='test')
    if args.okvqa_file == None:
        okvqa_filepath = data_path
    okvqa_test = pd.DataFrame.from_records(np.load(okvqa_filepath, allow_pickle=True)[1:])
    
    # Get predictions
    data = prediction_dataframe(model, data=okvqa_test)
    
    # Stratifications
    if args.stratify_by != None:
        condition = '' #TODO: implement stratification options
        data = data[condition]
            
        # Plot t-SNE
        if args.tsne:
            plot_TSNE(model, data=okvqa_test, label_by=args.stratify_by)
    
    # Performance report
    if args.performance_report:
        performance_report(data)
      

