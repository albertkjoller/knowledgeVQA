#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:14:28 2022

@author: s194253
"""

import sys
from pathlib import Path

import torch
from mmf.models import Qlarifais

import cv2

from mmexp.methods import *
from mmexp.utils.visualize import plot_example
from mmexp.utils.tools import image2tensor


def str_to_class(classname):
    return getattr(sys.modules['mmexp.methods'], classname)

if __name__ == "__main__":

    # obtain user input
    save_dir = input("Enter directory path of saved models ('save'-folder): ")
    model_name = input("Enter saved model filename: ")
    path_to_torch_cache = input("Enter the path to where your torch-cache folder is located (e.g. /work3/s194253):")

    model = Qlarifais.from_pretrained(f"{save_dir}/models/{model_name}", path_to_torch_cache)
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    
    # Specify answer vocab
    answer_vocab = model.processor_dict['answer_processor'].answer_vocab.word_list
    
    # initialize data variables
    old_img_name = None
    img_name, question = None, None

    # continue until user quits
    while question != 'quit()':
        print(f"\n{'-'*70}\n")

        # input image
        img_name = input("Enter image name from '../imgs/temp' folder (e.g. 'rain.jpg'): ")
        if old_img_name != None:
            cv2.destroyWindow(f"{old_img_name}")

        if img_name != 'quit()': # continues until user quits
            # load image
            img_path = Path(f"./../imgs/temp/{img_name}").as_posix()
            img = cv2.imread(img_path)  # open from file object

            # calculate the 50 percent of original dimensions
            width = int(img.shape[1] * 0.2)
            height = int(img.shape[0] * 0.2)

            # show image
            cv2.namedWindow(f"{img_name}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"{img_name}", width, height)
            cv2.imshow(f"{img_name}", img)
            cv2.waitKey(1)
        else:
            break

        # input question
        question = input("Enter question: ")
        answer = input("Enter ground truth answer (from answer list): ")
        explainability_method = str_to_class(input("Enter explainability method: "))

        # Process input
        image_tensor = image2tensor(img_path)
        category_id = model.processor_dict['answer_processor'].word2idx(answer)
                
        # TODO: make this general
        # Run explainability method
        saliency = explainability_method(model, image_tensor, category_id)
        
        plot_example(image_tensor, 
                     saliency, 
                     method='gradient', 
                     category_id=category_id,
                     answer_vocab=answer_vocab,
                     show_plot=True,
                     )
        

    # when loop is ended
    cv2.destroyAllWindows()