
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
    def from_pretrained(cls, model_name, path_to_torch_cache, *args, **kwargs):
        model = super().from_pretrained(model_name, *args, **kwargs)
        config = load_pretrained_model(model_name)["full_config"]
        OmegaConf.set_struct(config, True)
        return QlarifaisInterface(model, config, path_to_torch_cache)

    @classmethod
    def config_path(cls):
        # Relative to user dir root
        return "configs/models/qlarifais/defaults.yaml"

    # Do indirect path stuff with mmf
    def mmf_indirect(path):
        if os.path.exists(path):
            return path
        else:
            path = os.path.join(get_mmf_cache_dir(), "data/datasets", path)
            return path

    def build(self):

        # building general modules
        self.vision_module = build_image_encoder(self.config.image_encoder)
        self.language_module = build_text_encoder(self.config.text_encoder)
        self.fusion_module = build_fusion_module(self.config.fusion)
        self.classifier = build_classifier(self.config.classifier)

        # external knowledge
        self.graph_encoder = build_graph_encoder(self.config.graph_encoder)

        # attention
        if self.config.attention.use:
            # initiating attention module
            self.attention_module = build_attention_module(self.config.attention.params)

        #self.answer_processor = registry.get(self.config.datasets + "_answer_processor")
        #self.answer_vocab = self.answer_processor.answer_vocab
        #self.answer_vocab = registry.get(self.config.dataset_name + "_answer_processor").answer_vocab
        self.answer_vocab = VocabDict(mmf_indirect(config.vocab_file))
        self.embedded_answer_vocab = self.graph_encoder(self.answer_vocab.word_list)

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
            graph_features = self.graph_encoder(sample_list['tokens']) # [batch_size, g_dim]


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
        # average embedded annotator answer for type contrastive loss
        avg_embedded_answers  = self.graph_encoder(sample_list['answers'])
        if self.config.classifier.output_type == 'embeddings':
            prediction_scores = torch.nansum(logits.unsqueeze(dim=1) * self.embedded_answer_vocab, dim=2)
        else:
            prediction_scores = logits

        output = {"scores": logits, "output_type": self.config.classifier.output_type,
                  "avg_embedded_answers": avg_embedded_answers, 'prediction_scores': prediction_scores}

        return output
