#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 18:30:53 2022

@author: s194253
"""

import sys, os
import numpy as np
import pandas as pd
import torch

from mmf.models import Qlarifais

sys.path.append("..")
from mmexp.analyzer import *
from mmexp.utils.tools import paths_to_okvqa, str_to_class, fetch_test_embeddings

import argparse
import logging

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
        "--okvqa_file",
        help="path + filename of the okvqa-datafile, if not using the default \
        MMF one - e.g. if stratification on image objects is desired ",
        default=None,
    )        
    parser.add_argument(
        "--stratify_by",
        nargs='+',
        help="how to stratifiy the data. If not specified, all test-data is used. \
            Options: ['okvqa_categories', 'start_words']",
        default=None,
        #default=['start_words', 'okvqa_categories', 'question_length', 'answer_length', 'numerical_answers', 'visual_objects'],
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
    parser.add_argument(
        "--test",
        help="Whether to get predictions or not",
        default=True,
    )
        
    return parser.parse_args()

def init_logger(args):
    
    # Create logging directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Create and configure logger
    logging.basicConfig(filename=f"{args.save_path}/analysis.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
     
    # Setup info
    logger.info(f"\n{'-'*100}\n{'-'*100}\nRunning analyses for {args.model_dir.split('/')[-1]}\n{'-'*100}\n{'-'*100}\n")
                
    return logger

if __name__ == '__main__':
    
    # Get arguments
    args = get_args()
    
    # Initialize logger
    logger = init_logger(args)
    
    # Load model
    model = Qlarifais.from_pretrained(args.model_dir, args.torch_cache)
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model_name = args.model_dir.split("/")[-1]
    
    # Load data
    data_path, images_path = paths_to_okvqa(model, run_type='test')
    if args.okvqa_file == None:
        okvqa_filepath = data_path
        data = pd.DataFrame.from_records(np.load(okvqa_filepath, allow_pickle=True)[1:])
    else:
        # load data-investigation pickles (json)
        data = pd.read_json(args.okvqa_file)
    
    if args.test:    
        # Get predictions
        data = prediction_dataframe(model=model, data=data, report_dir=args.report_dir)
        embeddings = fetch_test_embeddings(model, args.pickle_path)
        
    # Performance report
    if args.performance_report:
        # Create performance report object
        performance_report = PerformanceReport(data, 
                                               embeddings,
                                               logger)
        # Call performance report
        performance_report.collect()
        logger = performance_report.logger
        
            
    # Stratifications
    if args.stratify_by != None:
        for strat_type in args.stratify_by:
            # Create stratification object
            stratified_object = Stratify(model, data, 
                                         by=strat_type, 
                                         pickle_path=args.pickle_path,
                                         )
            
            # Call stratified data
            stratified_data = stratified_object.data
            stratified_embeddings = stratified_object.embeddings
            
            # Compute t-SNE        
            if args.tsne:
                plot_TSNE(stratified_object, model_name=model_name, save_path=args.save_path)
            
            # Performance report
            if args.performance_report:
                performance_report = PerformanceReport(stratified_data,
                                                       stratified_embeddings,
                                                       logger)
                # Call performance report
                performance_report.collect(strat_type)
                logger = performance_report.logger
            
            

            
      

