import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn.functional as F

from captum_vqa.pytorch_resnet import resnet  # from pytorch-resnet
from captum_vqa.pytorch_vqa.model import Net, apply_attention, tile_2d_over_nd # from pytorch-vqa
from captum_vqa.pytorch_vqa.utils import get_transform # from pytorch-vqa

class ResNetLayer4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.r_model = resnet.resnet152(pretrained=True)
        self.r_model.eval()
        self.r_model.to(self.device)

    def forward(self, x):
        x = self.r_model.conv1(x)
        x = self.r_model.bn1(x)
        x = self.r_model.relu(x)
        x = self.r_model.maxpool(x)
        x = self.r_model.layer1(x)
        x = self.r_model.layer2(x)
        x = self.r_model.layer3(x)
        return self.r_model.layer4(x)

class VQA_Resnet_Model(Net):
    def __init__(self, embedding_tokens):
        super().__init__(embedding_tokens)
        self.resnet_layer4 = ResNetLayer4()

    def forward(self, v, q, q_len):
        q = self.text(q, list(q_len.data))
        v = self.resnet_layer4(v)

        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)

        a = self.attention(v, q)
        v = apply_attention(v, a)

        combined = torch.cat([v, q], dim=1)
        answer = self.classifier(combined)
        return answer
