#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 18:30:53 2022

@author: s194253
"""

import sys, os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import cm

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
        help="directory path to where the model-predictions are stored.",
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
        default='True',
    ) 
    parser.add_argument(
        "--performance_report",
        help="Prints a performance report on the data.",
        default='True',
    )
    parser.add_argument(
        "--plot_bars",
        help="Creates a bar plot from the stratified predicted scores. \
            Requires stratification flag.",
        default='True',
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
    
    # --model_dir /work3/s194262/save/models/optimized/baseline_ama --torch_cache /work3/s194253 --report_dir /work3/s194253/results/baseline_ama/reports --save_path /work3/s194253/results/baseline_ama --okvqa_file /work3/s194253/OKVQA_rich.json --stratify_by start_words okvqa_categories question_length answer_length numerical_answers num_visual_objects visual_objects_types --performance_report True
    
    # Get arguments
    args = get_args()
    args.performance_report = args.performance_report == 'True'
    args.tsne = args.tsne == 'True'
    args.plot_bars = args.plot_bars == 'True'

    
    # Load model
    model = Qlarifais.from_pretrained(args.model_dir, args.torch_cache)
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model_name = args.model_dir.split("/")[-1]
    
    # Initialize logger
    logger = init_logger(args)
    
    # Load data
    data_path, images_path = paths_to_okvqa(model, run_type='test')
    if args.okvqa_file == None:
        okvqa_filepath = data_path
        data = pd.DataFrame.from_records(np.load(okvqa_filepath, allow_pickle=True)[1:])
    else:
        # load data-investigation pickles (json)
        data = pd.read_json(args.okvqa_file)
    
    # Get predictions
    data = prediction_dataframe(model=model, data=data, report_dir=args.report_dir)
    embeddings = fetch_test_embeddings(model, args.report_dir)

    # Numberbatch
    Numberbatch_object = Numberbatch(model)
    
    # Performance report
    if args.performance_report == True:
        # Create performance report object
        performance_report = PerformanceReport(data, 
                                               embeddings,
                                               logger,
                                               Numberbatch_object)
        # Call performance report
        performance_report.collect()
        logger = performance_report.logger
        
        # Bar plot scores
        barplot_dict = defaultdict(dict)
        label = 'full dataset'
        barplot_dict[label] = defaultdict(dict)
        
        # Bar plot scores
        for key in performance_report.scores.keys():
            barplot_dict[label][key]['avg'] = performance_report.scores[key]
            barplot_dict[label][key]['CIs'] = performance_report.CIs[key]
            
    # Stratifications
    if args.stratify_by != None:
        for strat_type in args.stratify_by:
            logger.info(f"\n\n\n Stratifying by {strat_type}...\n")

            # Create stratification object
            stratified_object = Stratify(model, data, 
                                         by=strat_type, 
                                         pickle_path=args.report_dir,
                                         )
            
            # Compute t-SNE        
            if args.tsne == True:
                plot_TSNE(stratified_object, model_name=model_name, save_path=args.save_path)
            
            # Performance report
            if args.performance_report == True:
                
                # strat_labels
                barplot_dict = defaultdict(dict)
                strat_labels = stratified_object.data['stratification_label'].unique()
                for label in tqdm(strat_labels):

                    # Stratification indeces
                    strat_idxs = stratified_object.data.stratification_label == label
                    
                    # Stratify data + embeddings
                    stratified_data = stratified_object.data[strat_idxs].reset_index(drop=True)
                    stratified_embeddings = stratified_object.embeddings[:, strat_idxs]
                    
                    # Create performance report
                    performance_report = PerformanceReport(stratified_data,
                                                           stratified_embeddings,
                                                           logger,
                                                           Numberbatch_object)
                    
                    # Call performance report
                    performance_report.collect(f'{strat_type} = {label}')
                    logger = performance_report.logger
                    
                    # Bar plot scores
                    barplot_dict[label] = defaultdict(dict)
                    for key in performance_report.scores.keys():
                        barplot_dict[label][key]['avg'] = performance_report.scores[key]
                        barplot_dict[label][key]['CIs'] = performance_report.CIs[key]
                        
                if args.plot_bars == True:
                
                    # Change dict to dataframe for plotting
                    newdict = {(k1, k2):v2 for k1,v1 in barplot_dict.items() \
                               for k2,v2 in barplot_dict[k1].items()}  
                    df = pd.DataFrame([newdict[i] for i in sorted(newdict)],
                                      index=pd.MultiIndex.from_tuples([i for i in sorted(newdict.keys())]))
                
                    df['low'] = df.CIs.apply(lambda x: x[0])
                    df['high'] = df.CIs.apply(lambda x: x[1])
                                        
                    df = df.reset_index().rename(columns={'level_0': 'label', 'level_1': 'score'}).set_index(['label', 'score'])
                    df = df.unstack(level='label')
                    
                    df = df.rename(index={'vqa_acc': 'VQA Acc.', 'numberbatch_score': 'Numberbatch', 'acc': 'Accuracy'})
                    df = df.drop('Accuracy')

                    fig = plt.figure(dpi=400)
                    cmap = cm.get_cmap('tab20')
                    df['avg'].plot(kind='barh', xerr=df[['low','high']].T.values,
                                     width=0.88, cmap=cmap,
                                     figsize=(10,8), ax=plt.gca())
                    plt.title(f"Stratified by: {strat_type}", fontsize=15)
                    plt.legend(loc='lower right')
                    
                    os.makedirs(Path(args.save_path) / f'bars', exist_ok=True)
                    plt.savefig(Path(args.save_path) / f'bars/{strat_type}.png')                    
                    
                    
                    
                    
                
            
      

