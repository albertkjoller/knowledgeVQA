#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 10:03:18 2022

@author: s194253
"""

import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import json

from pathlib import Path

from pprint import pprint
import warnings
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class Numberbatch(nn.Module):
    """The generic class for graph networks
    Can be generically added to any other kind of network
    """

    def __init__(self, model, max_seq_length=128):
        super().__init__()
        self.model = model
        self.max_seq_length = max_seq_length
        
        # path to numberbatch-file
        numberbatch_path = Path(model.config.env.data_dir) / 'datasets/okvqa/defaults/graph/numberbatch-en.txt'
                    
        if os.path.exists(numberbatch_path):
            try:
                if self.numberbatch == dict():
                    pass
            except AttributeError:
                self.numberbatch = {}
                
                with open(numberbatch_path, 'rb') as f:
    
                    info = f.readlines(1)
                    lines, self.numberbatch_dim = (int(x) for x in info[0].decode('utf-8').strip("\n").split(" "))
    
                    for line in tqdm(f, total=lines):
                        l = line.decode('utf-8')
                        l = l.strip("\n")
    
                        # create embedding-dictionary
                        word = l.split(' ')[0]
                        embedding = np.array(list(map(float, l.split(' ')[1:])), dtype=np.float32)
                        self.numberbatch[word] = embedding
                        
            # Save numberbatch as a json object (faster loading)
            #json = json.dumps(self.numberbatch)
            #f = open("numberbatch-en.json","w")            
            #f.write(json)
            #f.close()

        else:
            pass
                    
    def conceptualize(self, tokenized_sentence):

        """
        Input:
        sentence (str): input sentence

        Output:
        concepts_found (set): the set of concepts in the sentence which are available in numberbatch
        """

        concepts_found = []
        start = 0
        while start < len(tokenized_sentence):
            for end in range(len(tokenized_sentence), start, -1):
                concept = tokenized_sentence[start:end]
                try:
                    self.numberbatch["_".join(concept)]
                    concepts_found.append("_".join(concept))
                    start += 1
                    break
                except KeyError:
                    if start == end - 1:
                        start += 1
                    else:
                        pass
        return concepts_found


    def get_embedding(self, text):
        # input can be batch with list containing tokens or list of strings

        batch_size = len(text)
        # initializing graph embeddings
        X = np.ones((batch_size, self.numberbatch_dim, self.max_seq_length))
        # todo: write other than batch?
        # looping tokens for each batch
        for batch, tokens in enumerate(text):
            # if input is a string it needs tokenization
            if type(tokens) == str: # i.e. text is not tokenized
                tokens = tokens.split(' ') #

            # if bert has tokenized the text
            if '[CLS]' and '[SEP]' in tokens:
                tokens.remove('[CLS]')
                tokens.remove('[SEP]')

            tokens = self.conceptualize(tokens) # if bert has tokenized
            # set to nan values as default
            X[batch] *= np.nan

            # if no token found create one empty numberbatch embedding
            if tokens == []:
                X[batch][:,0] = np.zeros(self.numberbatch_dim)

            for i, token in enumerate(tokens):
                # check if token is present in numberbatch
                try:
                    # extrach embeddings
                    X[batch][:, i] = self.numberbatch[token]
                except KeyError:
                    pass
                
        # average embeddings
        X = torch.from_numpy(np.nanmean(X, axis=2))
        # applying l2 norm to get a unit vector
        X = F.normalize(X)
        return X

class PerformanceReport:
    
    def __init__(self, data, embeddings, logger, Numberbatch_object):
        
        # Initialize class parameters
        self.data = data
        self.embeddings = embeddings
        self.logger = logger
        self.numberbatch = Numberbatch_object
        
        # Call metrics functions
        self.vqa_acc = self.compute_vqa_acc()
        self.acc = self.compute_acc()
        self.numberbatch_score = self.compute_numberbatch_score()
        self.AP_at_5 = self.compute_AP_at_k(K=5)
        
        # Plot barplots of scores
        self.plot_bars()
        
    def compute_vqa_acc(self, ):
        
        vqa_acc = lambda pred_ans, gt_ans: min(sum([pred_ans==ans for ans in gt_ans]) / 3, 1)
        scores = self.data.apply(lambda x: vqa_acc(x.prediction, x.answers), axis=1)
        return scores
    
    def compute_acc(self, ):
        
        acc = lambda pred_ans, gt_ans: pred_ans in gt_ans
        scores = self.data.apply(lambda x: acc(x.prediction, x.answers), axis=1)
        return scores
        
    def compute_numberbatch_score(self, ):
                
        def numberbatch_score(pred_embedding, gt_ans):
            cos = torch.nn.CosineSimilarity(dim=1)
            all_sims = cos(torch.tensor(pred_embedding).unsqueeze(0), self.numberbatch.get_embedding(gt_ans))
            return np.mean(all_sims.numpy())
        
        scores = []
        for i, row in self.data.iterrows():
            pred_embedding = self.embeddings[:, i]
            gt_ans = row.answers
            
            score = numberbatch_score(pred_embedding, gt_ans)
            scores.append(score)
            
        return pd.Series(scores)
        
    def compute_AP_at_k(self, K=5):
    
        # run AP loop
        scores = []
        for i, row in self.data.iterrows():            
            # define relevant items and recommended items
            relevant = np.unique(row.answers)
            recommended = row.topk
            
            # compute AP@5
            AP_at_K = 0
            for k in range(1, K+1):
                # precision@k score
                precision_at_k = sum([pred in relevant for pred in recommended[:k]]) / k
                
                AP_at_K += precision_at_k
            AP_at_K /= K
            
            # append to score list
            scores.append(AP_at_K)
        scores = pd.Series(scores)
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
        self.CIs['numberbatch_score'] = self.bootstrap_CI(self.numberbatch_score)
        self.CIs['AP@5'] = self.bootstrap_CI(self.AP_at_5)
        
        # Performance dict
        self.scores = {'vqa_acc': self.vqa_acc.mean(),
                       'acc': self.acc.mean(),
                       'numberbatch_score': self.numberbatch_score.mean(),
                       'mAP@5': self.AP_at_5.mean(),
                       }
        
        if stratification == 'full':
            self.logger.info(f"\n{'-'*100}\n Performance report on {stratification} testset\n{'-'*100}\n")
        else:
            self.logger.info(f"\n{'-'*100}\n Performance report on testset - stratified by {stratification}\n{'-'*100}\n")
                
        # Log size
        self.logger.info(f"\nNumber of test points = {len(self.data)}\n")
        
        # Print scores and CIs
        self.logger.info(f"Performance scores:\n{self.scores}\n")
        self.logger.info(f"Bootstrapped CIs:\n{self.CIs}\n")


        self.logger.info(f"\n{'-'*100}")
    
    def plot_bars(self,):
        raise NotImplementedError("PLOT BARPLOTS!")

        