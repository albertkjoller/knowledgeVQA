
# importing
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

from mmf.modules.layers import ReLUWithWeightNormFC



# Register the model for MMF, "concat_bert_tutorial" key would be used to find the model
@registry.register_model("pilot")
class Baseline(BaseModel):
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
        # Relative to user dir root
        return "configs/models/pilot/defaults.yaml"

    # Each method need to define a build method where the model's modules
    # are actually build and assigned to the model
    def build(self):


        self.vision_module = build_image_encoder(self.config.image_encoder)


        self.language_module = build_text_encoder(self.config.text_encoder)


        self.classifier = build_classifier_layer(self.config.classifier)

        # TODO: same as top-down but image_feat_dim hardcoded (not used)
        #self.non_linear_image = ReLUWithWeightNormFC(self.config.image_feat_dim, self.config.modal_hidden_size)



    # Each model in MMF gets a dict called sample_list which contains
    # all of the necessary information returned from the image
    def forward(self, sample_list):
        # Text input features will be in "input_ids" key
        text = sample_list["input_ids"]
        # Similarly, image input will be in "image" key
        image = sample_list["image"]

        # Get the text and image features from the encoders
        text_features = self.language_module(text)#[1]
        #print('here: ', len(text_features))
        #print('here: ', text_features[0].shape)


        image_features = self.vision_module(image)
        print(image_features.shape)

        # TODO: average pooling, lots of other options (top-down, sum, multi)
        #   - text-embedding and _operator has good example
        # doing it on dimensions 2 and 3
        image_features = torch.mean(image_features, dim = (2,3))


        # Flatten the embeddings before concatenation
        image_features = torch.flatten(image_features, start_dim=1)
        text_features = torch.flatten(text_features, start_dim=1)



        # Multiply the final features
        # TODO: (top down bottom up) haardman product
        combined = torch.cat([text_features, image_features], dim=1)


        # Pass final tensor to classifier to get scores
        logits = self.classifier(combined)
        # For loss calculations (automatically done by MMF
        # as per the loss defined in the config),
        # we need to return a dict with "scores" key as logits
        output = {"scores": logits}

        # MMF will automatically calculate loss
        return output











