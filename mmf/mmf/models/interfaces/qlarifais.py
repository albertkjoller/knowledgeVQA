# Copyright (c) Facebook, Inc. and its affiliates.

# Used for MMF internal models, hateful memes task,
# make predictions on raw images and texts

import os
import tempfile
import pandas as pd
from pathlib import Path
from typing import Type, Union

import pdb
import torch
import torchvision.datasets.folder as tv_helpers
from omegaconf import DictConfig
from mmf.common.sample import Sample, SampleList
from mmf.models.base_model import BaseModel
from mmf.utils.build import build_processors
from mmf.utils.download import download
from PIL import Image
from torch import nn


ImageType = Union[Type[Image.Image], str]
PathType = Union[Type[Path], str]
BaseModelType = Type[BaseModel]


class QlarifaisInterface(nn.Module):
    def __init__(self, model: BaseModelType, config: DictConfig, path_to_torch_cache: str):
        super().__init__()
        self.model = model
        self.config = config
        self.path_to_torch_cache = path_to_torch_cache
        self.init_processors()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def init_processors(self):
        config = self.config.dataset_config.okvqa 
        
        # update data path
        data_dir = pd.Series(config.data_dir.split("/"))
        torch_string_index = data_dir[data_dir == 'torch'].index[-1]
        data_dir = Path(self.path_to_torch_cache) / ("/").join(data_dir[torch_string_index:])
        
        extra_params = {"data_dir": data_dir.as_posix()}
        self.processor_dict = build_processors(config.processors, **extra_params)

    def classify(
        self,
        image: ImageType,
        text: str,
        top_k = None,
        explain = False,
    ):
        """Classifies a given image and text in it wrt. multi-class output defined 
        by MMF answer-txt-file.
        Image can be a url or a local path or you can directly pass a PIL.Image.Image
        object. Text needs to be a sentence containing a question.

        Args:
            image (ImageType): Image to be classified
            text (str): Text in the image

        Returns:
            {"label": '1930s', "confidence": 0.56}
        """
        sample = Sample()

        self.image_tensor = image = self.processor_dict["image_processor"](image).unsqueeze(0)
        self.image_tensor.requires_grad_(True)

        self.text = text = self.processor_dict["text_processor"]({"text": text})
        answer = self.processor_dict["answer_processor"]

        sample.image = self.image_tensor.squeeze(0)
        sample.text = text["text"]
        
        if "input_ids" in text:
            sample.update(text)

        sample_list = SampleList([sample])
        sample_list = sample_list.to(next(self.model.parameters()).device)
        sample_list['answers'] = 'empty'    
        
        output = self.model(sample_list)
        scores = nn.functional.softmax(output["prediction_scores"], dim=1)                      
        
        if top_k != None:
            confidence, indices = scores.topk(5, dim=1)

            top_k = [(p.item(), answer.idx2word(indices[0][idx].item())) for (idx, p) in enumerate(confidence[0])]        
            probs, answers = list(zip(*top_k))
            return probs, answers
        
        elif explain:
            return scores
            
        else:
            confidence, index = torch.max(scores, dim=1)
            return {"label": index.item(), "confidence": confidence.item()}
