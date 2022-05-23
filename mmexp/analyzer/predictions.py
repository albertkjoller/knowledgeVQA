#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 18:10:09 2022

@author: s194253
"""

import pandas as pd
import numpy as np

from pathlib import Path

from collections import Counter

from mmexp.utils.tools import load_predictions, fetch_test_embeddings


def prediction_dataframe(data, report_dir):
    # Get predictions
    predictions = load_predictions(Path(report_dir))
    predictions = predictions.rename(columns={'answer': 'prediction'})
    
    # Merge with test data
    data = data.merge(predictions, on='question_id')
    data = data[['question_id', 'question_tokens', 
                 'answers', 'prediction', 'topk',
                 'image_name']]    
    return data

class Stratify:
    
    def __init__(self, model, data, by, pickle_path):
        
        # Initialize class
        self.model = model
        self.data = data
        self.by = by
        self.embeddings = fetch_test_embeddings(model, pickle_path)

        self.stratify_data()
        
        if not self.temp_solution:
            # Category to index
            self.cat2idx = {cat: i for i, cat in enumerate(self.categories)}
    
    def stratify_data(self, ):
        
        # Temporary
        self.temp_solution = False
        
        if self.by == 'start_words':
            self.start_words()
            
        elif self.by == 'okvqa_categories':
            self.okvqa_categories()
            
        elif self.by == 'question_length':
            self.question_length()
            
        elif self.by == 'answer_length':
            self.answer_length()
        
        elif self.by == 'numerical_answers':
            self.numerical_answers()
        
        elif self.by == 'visual_objects':
            self.visual_objects()
                        
    def start_words(self, num_categories=10):
            
        # Start words
        start_words = self.data['question_tokens'].apply(lambda x: x[0])
        
        # Update data
        self.data['stratification_label'] = start_words
        self.categories = categories = list(zip(*Counter(start_words).most_common(num_categories)))[0]
                
        # Stratification condition
        strat_cond = self.data['stratification_label'].apply(lambda x: x in categories)
        
        # Restrict test data
        self.data = self.data[strat_cond].reset_index(drop=True)

        # Update embeddings
        self.embeddings = pd.DataFrame(self.embeddings).loc[:, start_words.apply(lambda x: x in categories)].to_numpy()
        
            
    def okvqa_categories(self, ):
        
        # TODO: implement based on data investigation pickles
        try:
            self.data['categories']
        except KeyError:
            self.temp_solution = True
            pass
            #raise AttributeError("Make sure that the file associated with the flag \
            #                     --okvqa_file contains an attribute called 'categories'")
    
    def question_length(self, ):
        
        # Question length
        question_length = self.data['question_tokens'].apply(lambda x: x.__len__())
        
        # TODO: change bins and category names
        bins, self.categories = [0, 5, 10, np.inf], ['short', 'intermediate', 'long']
        self.data['stratification_label'] = pd.cut(question_length, bins, labels=self.categories)
        

    def answer_length(self, ):
        self.temp_solution = True
        pass
        #raise NotImplementedError()
        
    def numerical_answers(self, ):
        
        has_number = lambda question: any(char.isdigit() for char in question)
        
        # TODO: implement something that accounts for multiple answers 
        answers_with_numbers = self.data['answers'].apply(lambda x: has_number(x))
        
        self.temp_solution = True
        pass
        #raise NotImplementedError()
        
    def visual_objects(self, ):
        self.temp_solution = True
        pass
        #raise NotImplementedError()
            