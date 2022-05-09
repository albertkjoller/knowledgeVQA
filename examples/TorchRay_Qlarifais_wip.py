#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:14:28 2022

@author: s194253
"""

from mmxai.interpretability.classification.torchray.extremal_perturbation.multimodal_extremal_perturbation import multi_extremal_perturbation, image2tensor
from mmxai.interpretability.classification.torchray.utils.visualize import plot_example

import torch
from mmf.models import Qlarifais


if __name__ == "__main__":
    
    image_path = '/zhome/b8/5/147299/Desktop/explainableVQA/imgs/temp/lynch.jpg'
    question = 'what is he famous for ?'
    model_name = 'ama_opt'
    path_to_torch_cache = '/work3/s194253'
    answer = 'movie'

    model = Qlarifais.from_pretrained(f"/work3/s194253/save/models/{model_name}", path_to_torch_cache)
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    
    # Process input
    image_tensor = image2tensor(image_path)
    category_id = model.processor_dict['answer_processor'].word2idx(answer)
    answer_vocab = model.processor_dict['answer_processor'].answer_vocab.word_list
    
    # Run extremal perturbation
    mask_, hist_, output_tensor, txt_summary, text_explaination = multi_extremal_perturbation(
        model,
        image_tensor,
        image_path,
        question,
        target=category_id, 
        max_iter=50,
        areas=[0.12],
    )
    
    # Visualize
    plot_example(image_tensor, 
                 mask_, 
                 method='extremal perturbation', 
                 category_id=category_id,
                 answer_vocab=answer_vocab,
                 show_plot=True,
                 )
    
    print("")
