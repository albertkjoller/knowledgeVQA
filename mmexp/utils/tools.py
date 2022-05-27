# -*- coding: utf-8 -*-

import os, sys, glob
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

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


def load_predictions(report_dir):
    
    for file in glob.glob((report_dir / '*.json').as_posix()):
        results = pd.read_json(file)
        break
    return results

def paths_to_okvqa(model, run_type='train'):
    # paths to data
    data_path = Path(model.config.dataset_config.okvqa.data_dir) / 'okvqa'
    images_path = data_path / 'defaults/images'

    if run_type == 'test':
        data_path = data_path / 'defaults/annotations/annotations/imdb_test.npy'
    elif run_type == 'train_val':
        data_path = data_path / 'defaults/annotations/annotations/imdb_trainval.npy'
    else:
        data_path = data_path / 'defaults/annotations/annotations/imdb_train.npy'
    return data_path, images_path

def fetch_test_embeddings(model, pickle_path):
    pickle_path = Path(pickle_path)
    
    try:
        # Try to load existing embedding
        with open(pickle_path / 'test_embeddings.npy', 'rb') as f:
            test_embeddings = np.load(f)
        print("Loaded embeddings successfully!")
    except FileNotFoundError:
        print("Creating test embeddings...")
        
        # paths to data
        data_path, images_path = paths_to_okvqa(model, run_type='test')
    
        # test dataset
        okvqa_test = pd.DataFrame.from_records(np.load(data_path, allow_pickle=True)[1:])
    
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
            test_embeddings[:, i] = outputs.cpu().detach().numpy()
        
        os.makedirs(pickle_path, exist_ok=True)
        with open(pickle_path / 'test_embeddings.npy', 'wb') as f:
            np.save(f, test_embeddings)
    
    return test_embeddings

def fetch_test_predictions(model, report_dir):
    report_dir = Path(report_dir)

    if os.path.exists(report_dir / 'test_predictions.csv'):
        test_predictions = pd.read_csv(report_dir / 'test_predictions.csv')
        print("Loaded predictions successfully!")
    else:
        print("Creating test predictions...")
        
        # paths to data
        data_path, images_path = paths_to_okvqa(model, run_type='test')
    
        # test dataset
        okvqa_test = pd.DataFrame.from_records(np.load(data_path, allow_pickle=True)[1:])
    
        # Initialize test embeddings
        test_predictions = np.ones((3, len(okvqa_test))).T * np.nan
        test_predictions = pd.DataFrame(test_predictions)
        test_predictions = test_predictions.rename(columns={0: 'question_id',
                                                            1: 'prediction',
                                                            2: 'topk'})
        
        # Obtain predictions
        for i, row in tqdm(okvqa_test.iterrows(), total=len(okvqa_test)):
            # load test image
            img_name = (images_path / row.image_name).as_posix() + '.jpg'
            image = tv_helpers.default_loader(img_name)
            
            # Get test question
            question = row.question_str
            
            # Get predicted embedding
            outputs = model.classify(image=image, text=question, top_k=5)
            outputs = outputs #.cpu().detach().numpy()
            
            # Write in dataframe
            test_predictions.loc[i, 'prediction'] = outputs[1][0]
            test_predictions.loc[i, 'topk'] = [list(zip(*outputs))]
            test_predictions.loc[i, 'question_id'] = int(row.question_id)
        
        try:
            test_predictions['question_id'] = test_predictions.question_id.astype(int)
        except pd.errors.IntCastingNaNError:
            pass
            
        os.makedirs(report_dir, exist_ok=True)
        test_predictions.to_csv(report_dir / 'test_predictions.csv', index=False)
        test_predictions = pd.read_csv(report_dir / 'test_predictions.csv')
        
    return test_predictions

def get_input(protocol_dir, protocol_name):
    
    input_file = Path(protocol_dir) / protocol_name
    if os.path.exists(input_file):
        with open(input_file, 'r') as f:
            info = f.read().splitlines()
            
        # Initialize parameters
        i = 0
        protocol_dict = defaultdict(dict)

        for string in info:
            if string == '':
                i += 1
            else:
                key, val = string.split(": ")
                protocol_dict[i][key] = val
                
    return protocol_dict
