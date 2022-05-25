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

from mmexp.utils.tools import load_predictions, fetch_test_embeddings, fetch_test_predictions

def prediction_dataframe(model, data, report_dir):
    # Get predictions
    predictions = fetch_test_predictions(model, report_dir)
    #predictions = load_predictions(Path(report_dir))
    #predictions = predictions.rename(columns={'answer': 'prediction'})
    
    # Merge with test data
    data = data.merge(predictions, on='question_id')
    data = data.drop(columns=['image_id', 'question_str', 'all_answers', 'feature_path'])
    return data

class Stratify:
    
    def __init__(self, model, data, by, pickle_path):
        
        # Initialize class parameters
        self.model = model
        self.data = data
        self.by = by
        
        # Fetch test embeddings
        self.embeddings = fetch_test_embeddings(model, pickle_path)

        # Stratify the data
        self.stratify_data()
        
        # Category to index
        self.cat2idx = {cat: i for i, cat in enumerate(self.categories)}
    
    def stratify_data(self, ):
        
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
        
        elif self.by == 'num_visual_objects':
            self.num_visual_objects()
            
        elif self.by == 'visual_object_types':
            self.visual_object_types()
                        
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
        # Question type mapping
        q_type = {'one': 'Vehicles and Transportation',
                  'two': 'Brands, Companies and Products',
                  'three': 'Objects, Material and Clothing',
                  'four': 'Sports and Recreation',
                  'five': 'Cooking and Food',
                  'six': 'Geography, History, Language and Culture',
                  'seven': 'People and Everyday life',
                  'eight': 'Plants and Animals',
                  'nine': 'Science and Technology',
                  'ten': 'Weather and Climate',
                  'other': 'Other',
                  }
        
        try:
            # Get question types
            question_types = self.data['question_type'].apply(lambda x: q_type[x])
            # Stratification labels
            self.categories = list(q_type.values())
            self.data['stratification_label'] = question_types
        except KeyError:   
            raise AttributeError("Make sure that the file associated with the flag \
                                 --okvqa_file contains an attribute called 'categories'")
    
    def question_length(self, ):
        # Question length
        question_length = self.data['question_tokens'].apply(lambda x: x.__len__())
        # Stratification labels        
        bins, self.categories = [0, 4, 7, 10, 13, np.inf], ['very short ( < 4)', 'short (4-7)', 'intermediate (7-10)', 'long (10-13)', 'very long (13 < )']
        self.data['stratification_label'] = pd.cut(question_length, bins, labels=self.categories)

    def answer_length(self, ):
        # Answer length
        answer_length = self.data['answers'].apply(lambda x: np.median([len(ans.strip(" ")) for ans in x]))
        # Stratification labels        
        bins, self.categories = [0, 1, 2, 4, 6, np.inf], ['very short (1)', 'short (2)', 'intermediate (3-4)', 'long (5-6)', 'very long (6 < )']
        self.data['stratification_label'] = pd.cut(answer_length, bins, labels=self.categories)
        
    def numerical_answers(self, ):
        # Function - string contains number
        has_number = lambda question: any(char.isdigit() for char in question)
        
        # Number of annotator answers with numerical content for each answer set
        num_numerical_ans = self.data['answers'].apply(lambda x: sum([has_number(ans) for ans in x]))
        
        # TODO: agree whether these boundaries are valid        
        # Stratification labels       
        bins, self.categories = [-np.inf, 3, 5, np.inf], ['not numerical ( < 3)', 'unclear (3-5)', 'numerical (5 < )']
        self.data['stratification_label'] = pd.cut(num_numerical_ans, bins, labels=self.categories)
        
    def num_visual_objects(self, ):
        # Get number of visual objects
        num_objects = self.data['image_objects'].apply(lambda x: x.__len__())
        # Stratification labels
        bins, self.categories = [-np.inf, 10, 20, 30, 40, 50, np.inf], ['< 10', '10-20', '20-30', '30-40', '40-50', '50 <']
        self.data['stratification_label'] = pd.cut(num_objects, bins, labels=self.categories)
    
    def visual_object_types(self, ):
        
        # Visual objects (from automated-object-removal)
        object_dict = {0: 'aeroplane',
                       1: 'bicycle',
                       2: 'bird',
                       3: 'boat',
                       4: 'bottle',
                       5: 'bus',
                       6: 'car',
                       7: 'cat', 
                       8: 'chair',
                       9: 'cow',
                       10: 'dining table',
                       11: 'dog',
                       12: 'horse',
                       13: 'motorbike',
                       14: 'person',
                       15: 'potted plant',
                       16: 'sheep',
                       17: 'sofa',
                       18: 'train',
                       19: 'tv/monitor',
                       }
        
        # single object images
        visual_objects = self.data['image_objects'].apply(lambda x: np.intersect1d(x, list(object_dict.values())))
        visual_objects = visual_objects[visual_objects.apply(lambda x: x.__len__() == 1)].apply(lambda x: x[0])
        
        # Update data
        self.data['stratification_label'] = visual_objects
        self.data = self.data.iloc[visual_objects.index, :].reset_index(drop=True)
        self.categories = list(object_dict.values())
        
        # Update embeddings
        self.embeddings = pd.DataFrame(self.embeddings).iloc[:, visual_objects.index].to_numpy()
            