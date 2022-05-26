#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 13:09:24 2022

@author: s194253
"""

import os
from pathlib import Path

import random

import cv2
import torch
import numpy as np

from mmexp.methods.automated_objects_removal_inpainter.src.config import Config
from mmexp.methods.automated_objects_removal_inpainter.src.edge_connect import EdgeConnect

class ObjectRemoval:
    
    def __init__(self, image_path, save_path, object4removal):
        
        self.object4removal = object4removal
        self.classes_dict = self.load_classes()
        self.object_id = self.object_to_remove()
        self.number_of_objects = 3 #int(input("Enter how many objects for removal: "))
        print("")
        
        self.input = image_path
        self.output = save_path #Path(f"./../imgs/removal_results/{self.object_name}").as_posix() 
        
        self.config = self.load_config()
        
    def load_classes(self,):
        classes_dict = {}
        with open("../mmexp/methods/automated_objects_removal_inpainter/segmentation_classes.txt") as f:
            for line in f:
               (val, key) = line.split(":")
               classes_dict[key.strip('\n').strip(' ')] = int(val)
               
        return classes_dict
    
    def object_to_remove(self, ):
        try:
            self.object_name = self.object4removal #input("Specify object to remove: ")
            return self.classes_dict[self.object_name]
        except KeyError:
            print("\nModel is not trained to remove this object... Try again!\n")
            self.object_to_remove()
    
    def load_config(self, mode=2):
        
        # config path
        config_path = '../mmexp/methods/automated_objects_removal_inpainter/checkpoints/config.yml'
        
        # load config file
        config = Config(config_path)
    
        # test mode
        config.MODE = 2
        config.MODEL = 3
        config.OBJECTS = [self.number_of_objects, self.object_id]
        config.SEG_DEVICE = 'cpu' #if args.cpu is not None else 'cuda'
        config.INPUT_SIZE = 256
    
        config.TEST_FLIST = self.input
        
        #if args.edge is not None:
        #    config.TEST_EDGE_FLIST = args.edge
        
        config.RESULTS = self.output 
        return config
    
    def remove_object(self,):
                
        # cuda visble devices
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in self.config.GPU)
    
        # init device
        if torch.cuda.is_available():
            self.config.DEVICE = torch.device("cuda")
            torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
        else:
            self.config.DEVICE = torch.device("cpu")
    
        # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
        cv2.setNumThreads(0)
    
        # initialize random seed
        torch.manual_seed(self.config.SEED)
        torch.cuda.manual_seed_all(self.config.SEED)
        np.random.seed(self.config.SEED)
        random.seed(self.config.SEED)
    
        # build the model and initialize
        model = EdgeConnect(self.config)
        model.load()
    
        # model test
        print('\nRemove object (running...)\n')
        model.test()