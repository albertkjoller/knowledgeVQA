import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import torch.nn.functional as F

from pytorch_resnet import resnet  # from pytorch-resnet
from pytorch_vqa.model import Net, apply_attention, tile_2d_over_nd # from pytorch-vqa
from pytorch_vqa.utils import get_transform # from pytorch-vqa

from models.models import VQA_Resnet_Model


### ---- Define setup ---- ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(Net().parameters(), lr=0.001, momentum=0.9)






print("hey")
