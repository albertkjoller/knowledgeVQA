#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 20:16:46 2022

@author: s194253
"""

import sys, os
from pathlib import Path

import torch
import cv2

from mmf.models import Qlarifais

sys.path.append("..")
from mmexp.utils.tools import str_to_class, get_input, load_image
from mmexp.utils.argument_wrapper import run_explainability, run_method

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
        "--protocol_dir",
        required=True,
        help="path to your torch cache directory, where the dataset is stored",
    )
    parser.add_argument(
        "--protocol_name",
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
        "--explainability_methods",
        nargs='+',
        help="list of explainability methods to be used, i.e. ['Gradient', 'GradCAM'].",
        default='Gradient',
    )  
    parser.add_argument(
        "--analysis_type",
        nargs='+',
        help="which analysis type to apply, e.g. 'OR', 'VisualNoise' or 'TextualNoise",
        default=None,
    )
    parser.add_argument(
        "--show_all",
        help="whether to save a combined image of the explainability methods",
        default=True,
    )    
    return parser.parse_args()


def init_logger(args):
    
    # Create logging directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Create and configure logger
    logging.basicConfig(filename=f"{args.save_path}/explainer.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
     
    # Setup info
    logger.info(f"\n{'-'*100}\n{'-'*100}\nRunning explainer-script for {args.model_dir.split('/')[-1]}\n{'-'*100}\n{'-'*100}\n")
                
    return logger

if __name__ == '__main__':
    
    # Get input
    args = get_args()
    protocol_dict = get_input(args.protocol_dir, args.protocol_name)

    # Initialize logger
    logger = init_logger(args)

    # Load model
    model = Qlarifais.from_pretrained(args.model_dir, args.torch_cache)
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model_name = args.model_dir.split("/")[-1]
    
    # Image directory
    imgs_dir = Path(args.protocol_dir) / 'imgs'
    
    # Loop through protocol items
    for i, input_set in protocol_dict.items():
        # Load input
        answer = input_set.get('A', None)
        question = input_set.get('Q', None)
        image_name = input_set.get('I', None).lower()
        remove_object = input_set.get('R', None)
        
        # Load image
        image_path = imgs_dir / image_name.split(".")[0]
        image = load_image((image_path / image_name).as_posix())
        
        # Add predictions to report
        prediction_str = f'PREDICTIONS: \n\nQuestion: "{question}"\nImage: {image_name}' + \
                            "\nPredicted outputs from the model:\n" 
        outputs = model.classify(image=image, text=question, top_k=5)
        for i, (prob, ans) in enumerate(zip(*outputs)):
            prediction_str += f"{i+1}) {ans} \t ({prob}\n)"
        logger.info(prediction_str)
        
        # Run explainability if answer is in answer vocab
        category_id = model.processor_dict['answer_processor'].word2idx(answer)
        if category_id == 0:
            logger.warning(f"\nThe ground truth '{answer}'not found in answer vocab - skipping...\n")
        else:
            logger.info("\nRunning explainability protocol...\n")
            
            # Run explainability methods
            for explainability_method in args.explainability_methods:             
                # Create exp-method object
                method = str_to_class(explainability_method)
                
                # Choose analysis types
                args.analysis_type.insert(0, 'Normal')
                for analysis_type in args.analysis_type:
                    if analysis_type == 'Normal':
                        mod_image = image
                        mod_question = question
                        
                    elif analysis_type == 'OR' and remove_object != None:
                        # Remove object                        
                        removal_path = Path(args.protocol_dir) / f'removal_results/{image_name.split(".")[0]}/{remove_object}'
                        if not os.path.exists(removal_path):
                            os.makedirs(removal_path)
                            
                            # remove objects from image
                            OR = str_to_class('OR')
                            OR_model = OR(image_path=image_path, 
                                          save_path=removal_path, 
                                          object4removal=remove_object)
                            OR_model.remove_object()
                        
                        # Load modified image
                        mod_image = load_image((removal_path / (".").join([image_name.split(".")[0], "png"])).as_posix())
                        mod_question = question

                    elif analysis_type == 'VisualNoise':
                        #image = 
                        pass
                    elif analysis_type == 'TextualNoise':
                        #question = 
                        pass
                    else:
                        if remove_object != None:
                            logger.warning(f"Analysis type - {analysis_type} - is not implemented...")
                            raise NotImplementedError(f"Analysis type - {analysis_type} - is not implemented...")
                            
                    save_name = Path(args.save_path) / f"explainability/{explainability_method}/{image_name.split('.')[0]}/{question.strip('?').replace(' ', '_').lower()}/{analysis_type.lower()}"
                    run_method(model, model_name, 
                               mod_image, image_name, 
                               mod_question, category_id, 
                               explainability_method,
                               save_path=save_name.as_posix(),
                               analysis_type=analysis_type,
                               )
                    
                    # Run explainability method for each image input
                    #run_explainability(model, model_name, 
                    #                   image, image_name, 
                    #                   question, category_id, 
                    #                   explainability_method,
                    #                   save_path=args.save_path
                    #                   )
                
                if args.show_all:
                    pass
            
            #explainer_img = cv2.imread(save_path_temp)
        
            
        
        
                    
        
        