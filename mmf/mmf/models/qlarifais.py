
import torch
import numpy as np
from pathlib import Path

from omegaconf import OmegaConf
from mmf.models.interfaces.qlarifais import QlarifaisInterface

from mmf.models.base_model import BaseModel
from mmf.common.registry import registry
from mmf.utils.checkpoint import load_pretrained_model
from mmf.utils.text import *

from mmf.utils.build import (
    build_image_encoder,
    build_text_encoder,
    build_graph_encoder,
    build_fusion_module,
    build_classifier,
    build_attention_module
    )

# mmf_run config='configs/experiments/baseline/mul.yaml' model=qlarifais dataset=okvqa run_type=train_val

@registry.register_model("qlarifais")
class Qlarifais(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.build()

    @classmethod
    def from_pretrained(cls, model_name, *args, **kwargs):
        model = super().from_pretrained(model_name, *args, **kwargs)
        config = load_pretrained_model(model_name)["full_config"]
        OmegaConf.set_struct(config, True)
        return QlarifaisInterface(model, config)

    @classmethod
    def config_path(cls):
        # Relative to user dir root
        return "configs/models/qlarifais/defaults.yaml"

    def build(self):

        # building general modules
        self.vision_module = build_image_encoder(self.config.image_encoder)
        self.language_module = build_text_encoder(self.config.text_encoder)
        self.fusion_module = build_fusion_module(self.config.fusion)
        self.classifier = build_classifier(self.config.classifier)

        # external knowledge
        self.graph_encoder = build_graph_encoder(self.config.graph_encoder)

        # used in classifier
        self.answer_vocab = registry.get(self.config.dataset_name + "_answer_processor").answer_vocab
        self.embedded_answer_vocab = self.graph_encoder({'tokens': [tokenize(sentence) for sentence in self.answer_vocab.word_list]})  # [batch_size, g_dim]
        self.num_not_top_k = len(self.embedded_answer_vocab) - int(self.config.classifier.params.top_k) # if classifier outputs embeddings
        # todo: tokenize better answer vocab
        #words = [self.answer_vocab.idx2word(idx) for idx, val in enumerate(torch.isnan(self.embedded_answer_vocab[:,0]).long()) if val == 1]
        #self.save_dir = Path(f'{self.config.cache_dir}/embeddings.pt')


        # attention
        if self.config.attention.use:
            # initiating attention module
            self.attention_module = build_attention_module(self.config.attention.params)

    def forward(self, sample_list):

        # --- QUESTION EMBEDDINGS ---
        # text input features will be in "input_ids" key
        question = sample_list["input_ids"]
        # get the text and image features from the encoders
        question_features = self.language_module(question)
        # IMAGE FEATURES
        image = sample_list["image"]
        image_features = self.vision_module(image) # [batch_size, i_dim, sqrt(max_features), sqrt(max_features)] # TODO: ?

        # --- GRAPH EMBEDDINGS ---
        if self.config.graph_encoder.use:
            graph_features = self.graph_encoder(sample_list) # [batch_size, g_dim]


        # --- ATTENTION ---
        if self.config.attention.use:
            # getting correct input shape
            image_features = image_features.flatten(2,3).permute(0, 2, 1) # [batch_size, num_features, i_dim]
            # extracting attention based on defined attention mechanism
            if self.config.attention.type == 'question_guided':
                attention = self.attention_module(image_features, question_features)
            if self.config.attention.type == 'graph_guided':
                attention = self.attention_module(image_features, graph_features)
            if self.config.attention.type == 'question_graph_guided':
                attention = self.attention_module(image_features, question_features, graph_features)
            # [batch_size, num_features, 1]
            # weighted average of image features
            image_features = (attention * image_features).sum(1)  # [batch_size, i_dim]
        # if not using attention
        else:
            if self.config.image_encoder.resize == 'average_pooling':
                # average pool K features of size 2048
                image_features = torch.mean(image_features, dim = (2,3)) # [batch_size, i_dim]


        # --- FUSION ---
        # type of fusion based on inputs
        if self.config.graph_encoder.use:
            fused_features = self.fusion_module(image_features, question_features, graph_features)
        else:
            fused_features = self.fusion_module(image_features, question_features)
        # [batch_size, answer_vocab_dim]

        # --- CLASSIFICATION ---
        # embeddings
        logits = self.classifier(fused_features)

        #torch.save(embeddings, self.save_dir)
        output = {'output_type': self.config.classifier.output_type, 'scores': logits}

        return output


class read_object:
    def __init__(self, embedding):
        self.embedding = embedding

    def __get__(self, instance, owner):
        self.instance
    @property
    def get_emb(self):
        return self.embedding

