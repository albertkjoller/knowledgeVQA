# -*- coding: utf-8 -*-

import os, sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2
from PIL import Image
from torchray.utils import imsc

import torch
from torchvision import transforms
import torchvision.datasets.folder as tv_helpers

from mmf.utils.download import download

def image_loader(old_img_name):
    # input image
    img_name = input("Enter image name from '../imgs/temp/' folder (e.g. 'rain/rain.jpg'): ")
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


def image2tensor(image_path):
    # convert image to torch tensor with shape (1 * 3 * 224 * 224)
    img = Image.open(image_path)
    p = transforms.Compose([transforms.Scale((224, 224))])

    img, i = imsc(p(img), quiet=False)
    return torch.reshape(img, (1, 3, 224, 224))



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


def fetch_test_embeddings(model):
    pickle_path = Path(model.config.env.save_dir)
    try:
        # Try to load existing embedding
        with open(pickle_path / 'test_embeddings.npy', 'rb') as f:
            test_embeddings = np.load(f)
        print("Loaded embeddings successfully!")
    except FileNotFoundError:
        print("Creating test embeddings...")
        
        # paths to data
        data_path = Path(model.config.dataset_config.okvqa.data_dir) / 'okvqa'
        test_data_path = data_path / 'defaults/annotations/annotations/imdb_test.npy'
        images_path = data_path / 'defaults/images'
    
        # test dataset
        okvqa_test = pd.DataFrame.from_records(np.load(test_data_path, allow_pickle=True)[1:])
    
        # Initialize test embeddings
        test_embeddings = np.zeros((300, len(okvqa_test)))
    
        # Obtain predictions
        for i in tqdm(range(okvqa_test.__len__())):
            # load test image
            img_name = (images_path / okvqa_test.image_name[i]).as_posix() + '.jpg'
            image = tv_helpers.default_loader(img_name)
            
            # Get test question
            question = okvqa_test.question_str[i]
            
            # Get predicted embedding
            outputs = model.classify(image=image, text=question, embedding_output=True)
            test_embeddings[:, i] = outputs.detach().numpy()
            
        with open(pickle_path / 'test_embeddings.npy', 'wb') as f:
            np.save(f, test_embeddings)
    
    return test_embeddings