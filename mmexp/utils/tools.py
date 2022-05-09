# -*- coding: utf-8 -*-

import torch

from PIL import Image
from torchray.utils import imsc
from torchvision import transforms


def image2tensor(image_path):
    # convert image to torch tensor with shape (1 * 3 * 224 * 224)
    img = Image.open(image_path)
    p = transforms.Compose([transforms.Scale((224, 224))])

    img, i = imsc(p(img), quiet=False)
    return torch.reshape(img, (1, 3, 224, 224))
