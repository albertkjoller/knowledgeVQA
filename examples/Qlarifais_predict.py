#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:14:28 2022

@author: s194253
"""

from pathlib import Path

import torch
from mmf.models import Qlarifais


if __name__ == "__main__":

    import cv2

    # obtain user input
    save_dir = input("Enter directory path of saved models ('save'-folder): ")
    model_name = input("Enter saved model filename: ")

    model = Qlarifais.from_pretrained(f"{save_dir}/models/{model_name}")
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    
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


        # get predictions and show input
        topk = 5
        outputs = model.classify(image=img_path, text=question, top_k=topk)
        old_img_name = img_name

        # print answers and probabilities
        print(f'\nQuestion: "{question}"')
        print("\nPredicted outputs from the model:")
        for i, (prob, answer) in enumerate(zip(*outputs)):
            print(f"{i+1}) {answer} \t ({prob})")

    # when loop is ended
    cv2.destroyAllWindows()