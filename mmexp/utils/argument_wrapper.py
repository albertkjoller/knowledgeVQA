#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 19:11:06 2022

@author: s194253
"""

import sys, os
from pathlib import Path
import matplotlib.pyplot as plt

from mmexp.methods import *
from mmexp.utils.visualize import plot_example


import tempfile
from mmf.utils.download import download
import torchvision.datasets.folder as tv_helpers

def load_image(img_path):
    
    if isinstance(img_path, str):
        if img_path.startswith("http"):
            temp_file = tempfile.NamedTemporaryFile()
            download(img_path, *os.path.split(temp_file.name), disable_tqdm=True)
            image = tv_helpers.default_loader(temp_file.name)
            temp_file.close()
        else:
            image = tv_helpers.default_loader(img_path)
            
    return image

def str_to_class(classname):
    return getattr(sys.modules['mmexp.methods'], classname)


def run_explainability(model, model_name, image, img_name, question, category_id, explainability_method):
        
    # Specify answer vocab
    answer_vocab = model.processor_dict['answer_processor'].answer_vocab.word_list
    
    # Specify save path
    save_path = f"./../imgs/explainability/{model_name}/{explainability_method}/{img_name.split('/')[0]}/{question}"

    """
    # Run explainability method
    if explainability_method == 'MMEP':
        
        method = str_to_class(explainability_method)
        saliency, hist, x, summary, conclusion = method(model,
                                                        image_tensor,
                                                        image,
                                                        question,
                                                        category_id,
                                                        max_iter=50,
                                                        )
        plot_example(model.image_tensor, 
                     saliency, 
                     method=explainability_method, 
                     category_id=category_id,
                     answer_vocab=answer_vocab,
                     show_plot=False,
                     save_path=save_path + '.png',
                     )
    """  
    if explainability_method == 'MMGradient':
        method = str_to_class(explainability_method)
        saliency = method(model, 
                          image,
                          question,
                          category_id,
                          )
        
        # visualize gradient map
        plot_example(model.image_tensor, 
                     saliency, 
                     method=explainability_method, 
                     category_id=category_id,
                     answer_vocab=answer_vocab,
                     show_plot=False,
                     save_path=save_path + '.png',
                     )
        save_path = save_path + '.png'
        
    elif explainability_method == 'MMGradCAM':
        method = str_to_class(explainability_method)
        saliency = method(model, 
                          image,
                          question,
                          category_id,
                          )
        
        # visualize gradient map
        plot_example(model.image_tensor, 
                     saliency, 
                     method=explainability_method, 
                     category_id=category_id,
                     answer_vocab=answer_vocab,
                     show_plot=False,
                     save_path=save_path + '.png',
                     )
        save_path = save_path + '.png'
    
    
    
    
    elif explainability_method == 'MMGradientOR':
    
        # get gradient of original image
        saliency_orig = MMGradient(model, 
                                   image,
                                   question,
                                   category_id,
                                   )
        # visualize gradient map
        plt.subplot(211)
        plot_example(model.image_tensor, 
                     saliency_orig, 
                     method=explainability_method, 
                     category_id=category_id,
                     answer_vocab=answer_vocab,
                     show_plot=False,
                     save_path=save_path + '.png',
                     )
        
        # remove objects from image
        method = str_to_class(explainability_method)
        OR_model = method(img_name)
        OR_model.remove_object()
        
        # Load new image
        img_path = Path(f"./../imgs/removal_results/{OR_model.object_name}/{img_name.split('/')[-1]}").as_posix()
        modified_image = load_image(img_path)

        saliency_modified = MMGradient(model, 
                                       modified_image,
                                       question,
                                       category_id,
                                       )
        # visualize gradient map
        plt.subplot(212)
        plot_example(model.image_tensor, 
                     saliency_modified, 
                     method=explainability_method, 
                     category_id=category_id,
                     answer_vocab=answer_vocab,
                     show_plot=False,
                     save_path=save_path + f'_removed_{OR_model.object_name}.png',
                     )
        save_path = [save_path + '.png', save_path + f'_removed_{OR_model.object_name}.png']
    
    elif explainability_method == 'MMGradCAM-OR':
    
        # get gradient of original image
        saliency_orig = MMGradient(model, 
                                   image,
                                   question,
                                   category_id,
                                   )
        # visualize gradient map
        plt.subplot(211)
        plot_example(model.image_tensor, 
                     saliency_orig, 
                     method=explainability_method, 
                     category_id=category_id,
                     answer_vocab=answer_vocab,
                     show_plot=False,
                     save_path=save_path + '.png',
                     )
        
        # remove objects from image
        method = str_to_class(explainability_method)
        OR_model = method(img_name)
        OR_model.remove_object()
        
        # Load new image
        img_path = Path(f"./../imgs/removal_results/{OR_model.object_name}/{img_name.split('/')[-1]}").as_posix()
        modified_image = load_image(img_path)

        saliency_modified = MMGradient(model, 
                                       modified_image,
                                       question,
                                       category_id,
                                       )
        # visualize gradient map
        plt.subplot(212)
        plot_example(model.image_tensor, 
                     saliency_modified, 
                     method=explainability_method, 
                     category_id=category_id,
                     answer_vocab=answer_vocab,
                     show_plot=False,
                     save_path=save_path + f'_removed_{OR_model.object_name}.png',
                     )
        save_path = [save_path + '.png', save_path + f'_removed_{OR_model.object_name}.png']
        
        
    return save_path