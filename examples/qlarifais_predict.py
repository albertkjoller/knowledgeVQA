#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:14:28 2022

@author: s194253
"""

from pathlib import Path

import torch
from mmf.models import Qlarifais
import cv2

import os
import tempfile
from typing import Type, Union
from mmf.utils.download import download
import torchvision.datasets.folder as tv_helpers

if __name__ == "__main__":

    def image_loader(old_img_name):
        # input image
        img_name = input("Enter image name from '../imgs/temp' folder (e.g. 'rain.jpg'): ")
        if old_img_name != None:
            cv2.destroyWindow(f"{old_img_name}")
    
        if img_name != 'quit()': # continues until user quits
            # load image
            img2None = False
            img_path = Path(f"./../imgs/temp/{img_name}").as_posix()
            img = cv2.imread(img_path)  # open from file object
            
            if type(img) == type(None):
                img = cv2.imread(Path("./../imgs/temp/fail.png").as_posix())
                img2None = True       
            
            # show image
            cv2.namedWindow(f"{img_name}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"{img_name}", 600, 300)
            cv2.imshow(f"{img_name}", img)
            cv2.waitKey(1)
            
            if img2None:
                img = None
            
            return img_name, img_path, img
        else:
            return "quit()", None, None
        
        
    # obtain user input
    save_dir = input("Enter directory path of saved models ('save'-folder): ")
    model_name = input("\nEnter saved model filename: ")
    path_to_torch_cache = input("\nEnter the path to where your torch-cache folder is located (e.g. /work3/s194253):")

    model = Qlarifais.from_pretrained(f"{save_dir}/models/{model_name}", path_to_torch_cache)
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    
    # initialize data variables
    old_img_name = None
    img_name, question = None, None

    # continue until user quits
    while question != 'quit()':
        img = None

        print(f"\n{'-'*70}\n")
        while type(img) == type(None):
            
            img_name, img_path, img = image_loader(old_img_name)
            if img_name == 'quit()':
                break
            elif type(img) == type(None):
                print("\nImage-file does not exist - try again...\n")
            
            # update img_name
            old_img_name = img_name
        if img_name == 'quit()':
            cv2.destroyAllWindows()
            break
        
        else:
            # input question
            question = input("Enter question: ")
            
            if question == 'quit()':
                break
            else:
                if isinstance(img_path, str):
                    if img_path.startswith("http"):
                        temp_file = tempfile.NamedTemporaryFile()
                        download(img_path, *os.path.split(temp_file.name), disable_tqdm=True)
                        image = tv_helpers.default_loader(temp_file.name)
                        temp_file.close()
                    else:
                        image = tv_helpers.default_loader(img_path)
                
                # get predictions and show input
                topk = 5
                outputs = model.classify(image=image, text=question, top_k=topk)
        
                # print answers and probabilities
                print(f'\nQuestion: "{question}"')
                print("\nPredicted outputs from the model:")
                for i, (prob, answer) in enumerate(zip(*outputs)):
                    print(f"{i+1}) {answer} \t ({prob})")
             
    # when loop is ended
    cv2.destroyAllWindows()