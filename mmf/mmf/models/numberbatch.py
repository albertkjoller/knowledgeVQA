

import torch

from mmf.models.base_model import BaseModel
from mmf.common.registry import registry
from mmf.utils.build import (
    build_classifier_layer,
    build_image_encoder,
    build_text_encoder,
)

import gzip
from tqdm import tqdm
from typing import List

@registry.register_model("Numberbatch")
class Numberbatch(BaseModel):
    
    def __init__(self, config):
        
        # Path to numberbatch embeddings - download from here:
        # https://github.com/commonsense/conceptnet-numberbatch
        
        #TODO: fix this as an input in the config
        self.numberbatch_filepath = '/work3/s194253/numberbatch-en-19.08.txt.gz'
        
        super().__init__(config)
        self.build()

    @classmethod
    def config_path(cls):
        return "configs/models/xxx/defaults.yaml"

    def build(self):

        self.vision_module = build_image_encoder(self.config.image_encoder)
        self.language_module = build_text_encoder(self.config.text_encoder)
        
        #TODO: check the structure of config.graph_encoder - is it a dict? then this should
        # be fed to the function below
        self.build_graph_encoder(self.config.graph_encoder) # implicitly builds graph_module
        self.classifier = build_classifier_layer(self.config.classifier)

    def build_graph_encoder(self, ): #TODO: Specify filename in config?

        self.numberbatch = {} 
        with gzip.open(self.numberbatch_filepath, 'rb') as f:
            info = f.readlines(1)
            lines, self.dim = (int(x) for x in info[0].decode('utf-8').strip("\n").split(" "))

            for line in tqdm(f, total=lines):
                l = line.decode('utf-8')
                l = l.strip("\n")
                
                # create tensor-dictionary
                word = l.split(' ')[0]
                tensor = torch.tensor(list(map(float, l.split(' ')[1:])), dtype=torch.float32)
                self.numberbatch[word] = tensor

    def graph_module(self, question_tokens: List[str]):
        assert type(question_tokens[0]) != list, "Ensure that token input is a list of strings and not a list of a list!"

        X = torch.ones((self.dim, question_tokens.__len__())) * torch.nan
        for i, token in enumerate(question_tokens):
            try:
                tensor = self.numberbatch[token]
                X[:, i] = tensor        
            except KeyError:
                pass
        return X.nanmean(axis=1)
    
    def forward(self, sample_list):
        # Question input and embeddings
        text = sample_list['input_ids']
        text_features = self.language_module(text)
        text_features = torch.flatten(text_features, start_dim=1)

        # Image input and features
        image = sample_list['image']
        image_features = self.vision_module(image)
        
        #TODO: explain this
        image_features = torch.mean(image_features, dim = (2,3)) # to add average pooling based on attention

        # Flatten the embeddings before concatenation
        image_features = torch.flatten(image_features, start_dim=1)

        # External knowledge representation
        q_tokens = sample_list['tokens']
        graph_features = self.graph_module(text) #TODO: check this
        
        # Combine features
        fusion = torch.cat([text_features, image_features, graph_features], dim=1)
        #TODO: something with asserting that all features are on same scaling? mean subtraction?

        # Pass final tensor to classifier to get scores
        logits = self.classifier(fusion)

        # Return dict of logits with 'scores' as key for loss calculation
        output = {"scores": logits}
        return output






