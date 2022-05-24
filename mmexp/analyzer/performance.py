#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 10:03:18 2022

@author: s194253
"""

import numpy as np

from pprint import pprint
import warnings
import logging

class PerformanceReport:
    
    def __init__(self, data, embeddings, logger):
        
        # Initialize class parameters
        self.data = data
        self.logger = logger
        
        # Call metrics functions
        self.vqa_acc = self.compute_vqa_acc()
        self.acc = self.compute_acc()
        self.numberbatch_score = self.compute_numberbatch_score()
        self.mir = self.compute_mean_inverse_rank()
        
    def compute_vqa_acc(self, ):
        
        vqa_acc = lambda pred_ans, gt_ans: min(sum([pred_ans==ans for ans in gt_ans]) / 3, 1)
        scores = self.data.apply(lambda x: vqa_acc(x.prediction, x.answers), axis=1)
        return scores
    
    def compute_acc(self, ):
        
        acc = lambda pred_ans, gt_ans: pred_ans in gt_ans
        scores = self.data.apply(lambda x: acc(x.prediction, x.answers), axis=1)
        return scores
        
    def compute_numberbatch_score(self, ):
        # TODO: implement
        self.logger.warning("NotImplementetError: Add Numberbatch score")
        
    def compute_mean_inverse_rank(self, ):
        # TODO: implement
        self.logger.warning("NotImplementedError: Add mean inverse rank")
        scores = [1]
        return scores
    
    def bootstrap_CI(self, scores, nbr_draws=1000):
        
        means = np.zeros(nbr_draws)
        scores = np.array(scores)
    
        for n in range(nbr_draws):
            indices = np.random.randint(0, len(scores), len(scores))
            scores_temp = scores[indices] 
            means[n] = np.nanmean(scores_temp)
    
        return [np.nanpercentile(means, 2.5),np.nanpercentile(means, 97.5)]

    def collect(self, stratification='full'):

        # Combine metrics in a table / report    
        self.CIs = {}
    
        # Compute bootstrapped confidence intervals
        self.CIs['vqa_acc'] = self.bootstrap_CI(self.vqa_acc)
        self.CIs['acc'] = self.bootstrap_CI(self.acc)
        #self.CIs['numberbatch_score] = self.bootstrap_CI(self.numberbatch_score)
        #self.CIs['mir'] = self.bootstrap_CI(self.mir)
        
        # Performance dict
        self.scores = {'vqa_acc': self.vqa_acc.mean(),
                       'acc': self.acc.mean(),
                       'numberbatch_score': self.numberbatch_score.mean(),
                       'mean_inverse_rank': self.mir.mean(),
                       }
        
        if stratification == 'full':
            self.logger.info(f"\n{'-'*50}\n Performance report on {stratification} testset\n{'-'*50}\n")
        else:
            self.logger.info(f"\n{'-'*50}\n Performance report on testset - stratified by {stratification}\n{'-'*50}\n")
            
        # Print scores and CIs
        self.logger.info(f"\nPerformance scores:\n")
        self.logger.info(self.scores)
        self.logger.info(f"\n\nBootstrapped CIs:\n")
        self.logger.info(self.CIs)
        self.logger.info(f"\n{'-'*50}")

        