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
from mmexp.utils.argument_wrapper import run_explainability

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
        help="list of explainability methods to be used, i.e. ['Gradient', 'OR-Gradient']. \
            Requires --explain flag.",
        default='Gradient',
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
        image_name = input_set.get('I', None)
        remove_object = input_set.get('R', None)
        
        # Load image
        image_path = imgs_dir / image_name.split(".")[0] / image_name
        image = load_image(image_path.as_posix())
        
        # Add predictions to report
        outputs = model.classify(image=image, text=question, top_k=5)
        logger.info(f'\nQuestion: "{question}\nImage: {image_name}"')
        logger.info("\nPredicted outputs from the model:\n")
        for i, (prob, ans) in enumerate(zip(*outputs)):
            logger.info(f"{i+1}) {answer} \t ({prob})")
        
        # Run explainability if answer is in answer vocab
        category_id = model.processor_dict['answer_processor'].word2idx(answer)
        if category_id == 0:
            logger.warning(f"\nThe ground truth '{answer}'not found in answer vocab - skipping...\n")
        else:
            logger.info(f"\nRunning explainability protocol...\n")
            
            # Run explainability methods
            for explainability_method in args.explainability_methods:
                method = str_to_class(explainability_method)
                    
            save_path_temp = run_explainability(model, model_name, 
                                                image, image_name, 
                                                question, category_id, 
                                                explainability_method,
                                                )
            explainer_img = cv2.imread(save_path_temp)
            
# =============================================================================
#             
#         else:
#             # Read images
#             img1 = cv2.imread(save_path[0]) #orig
#             img2 = cv2.imread(save_path[1]) # exp_map
#             
#             explainer_img = np.concatenate((img1, img2), axis=0)
#             cv2.imwrite(save_path[0].split(".png")[0] + f'_{checkpoint_to_use}_full.png', explainer_img)
#             
#             # show explainability image
#             cv2.namedWindow(f"{explainability_method}", cv2.WINDOW_NORMAL)
#             cv2.resizeWindow(f"{explainability_method}", 600, 300)
#             cv2.imshow(f"{explainability_method}", explainer_img)
#             cv2.waitKey(1)
# =============================================================================
        
            
        
        
                    
        
        