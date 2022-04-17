

import torch
# All model using MMF need to inherit BaseModel
from mmf.models.base_model import BaseModel
# registry is need to register the dataset or our new model so as to be MMF discoverable
from mmf.common.registry import registry
# Builder methods for image encoder and classifier
from mmf.utils.build import (
    build_classifier_layer,
    build_image_encoder,
    build_text_encoder,
)

import gzip
from tqdm import tqdm
import numpy as np

'''
run command:

mmf_run config='configs/experiments/external_knowledge/grids.yaml' model=external_knowledge dataset=okvqa run_type=train_val
'''

# Register the model for MMF, "concat_bert_tutorial" key would be used to find the model
@registry.register_model("Temporary")
class Temporary(BaseModel):
    # All models in MMF get first argument as config which contains all
    # of the information you stored in this model's config (hyperparameters)
    def __init__(self, config):
        # This is not needed in most cases as it just calling parent's init
        # with same parameters. But to explain how config is initialized we
        # have kept this
        super().__init__(config)
        self.build()

    # This classmethod tells MMF where to look for default config of this model
    @classmethod
    def config_path(cls):
        # Relative to user dir root #TODO: CHECK HERE
        return "configs/models/xxx/defaults.yaml"

    # Each method need to define a build method where the model's modules
    # are actually build and assigned to the model
    def build(self):

        self.vision_module = build_image_encoder(self.config.image_encoder)
        self.language_module = build_text_encoder(self.config.text_encoder)
        #TODO: check the structure of config.graph_encoder - is it a dict? then this should
        # be fed to the function below
        self.build_graph_encoder(self.config.graph_encoder) # implicitly builds graph_module
        self.classifier = build_classifier_layer(self.config.classifier)

    def build_graph_encoder(self, ): #TODO: check if config-input is needed

        numberbatch = {} #TODO: Specify filename in config?
        with gzip.open(filename, 'rb') as f:
            info = f.readlines(1)
            lines, dim = (int(x) for x in info[0].decode('utf-8').strip("\n").split(" "))

            for line in tqdm(f, total=lines):
                l = line.decode('utf-8')

                #TODO: Check if this is equal to the one below --> arr = ...
                l = l.strip("\n")
                word = l.split(' ')[0]
                arr = torch.tensor(list(map(float, l.split(' ')[1:])), dtype=torch.float32)

                #l_split = l.strip("\n").split(" ")
                #arr = np.fromstring((" ").join(l_split[1:]), dtype=float, sep=" ")
                #word = l_split[0]
                numberbatch[word] = arr

        # numberbatch dict
        self.numberbatch = numberbatch

    def graph_module(self, text):
        #TODO: Check if text is tokenized
        q_embedding = []
        for token in text:
            try:
                arr = self.numberbatch[token]
                #TODO: change to torch
                q_embedding.append(arr)
            except KeyError:
                pass

        #TODO: change to torch
        q_embedding = np.mean(q_embedding, axis=0)
        return q_embedding

    def forward(self, sample_list):

        # Question input and embeddings
        text = sample_list["input_ids"]
        text_features = self.language_module(text)#[1]
        text_features = torch.flatten(text_features, start_dim=1)

        # Image input and features
        image = sample_list["image"]
        image_features = self.vision_module(image)

        # TODO: average pooling, lots of other options (top-down, sum, multi)
        #   - text-embedding and _operator has good example
        # Doing it on dimensions 2 and 3 and keep 2048
        image_features = torch.mean(image_features, dim = (2,3)) # to add average pooling based on attention

        # Flatten the embeddings before concatenation
        image_features = torch.flatten(image_features, start_dim=1)

        # External knowledge
        # initialize for graph module
        sample_list["q_encoded"] = text # dim 128
        graph_features = self.graph_module(text) #TODO: check if this is correct - text must be tokens

        # combining features
        #TODO: something with asserting that all features are on same scaling? mean subtraction?
        fusion = torch.cat([text_features, image_features, graph_features], dim=1)

        # Pass final tensor to classifier to get scores
        logits = self.classifier(fusion)

        # For loss calculations (automatically done by MMF
        # as per the loss defined in the config),
        # we need to return a dict with "scores" key as logits
        output = {"scores": logits}

        # MMF will automatically calculate loss
        return output






