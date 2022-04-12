

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
from mmf.modules.prior import load_priors

'''
run command:
# example
mmf_run config='configs/experiments//defaults.yaml' model=qlarifais dataset=okvqa run_type=train_val
'''

# Register the model for MMF, "concat_bert_tutorial" key would be used to find the model
@registry.register_model("qlarifais")
class Qlarifais(BaseModel):
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
        return "configs/models/qlarifais/defaults.yaml"

    # Each method need to define a build method where the model's modules
    # are actually build and assigned to the model
    def build(self):


        self.vision_module = build_image_encoder(self.config.image_encoder)

        self.language_module = build_text_encoder(self.config.text_encoder)

        self.classifier = build_classifier_layer(self.config.classifier)


        # if model uses external knowledge
        if self.config.graph_module.use:
            # Import graph network module
            self.graph_module = GraphNetworkModule(self.config.graph_module)

            # graph logits
            if self.config.graph_logit_mode == "in_graph":
                # Logits is already computed
                assert self.config.graph_module.output_type == "graph_prediction"

            elif self.config.graph_logit_mode == "logit_fc":
                # Compute logits from single hidden layer
                self.graph_logit_fc = nn.Linear(self.config.graph_module.node_hid_dim, self.config.num_labels)

            # whether to add or concat features (where 'add' meaning we want the num_labels size from answer vocab)
            # Answer indices not in graph if we are adding features
            if self.config.output_combine == "add":
                # Output order should be ans
                assert self.config.graph_module.output_order == "ans"
                self.missing_ans_inds = torch.LongTensor(self.config.num_labels).fill_(1)
                # Now any index stil set to 1 is missing from graph
                self.missing_ans_inds[self.graph_module.index_in_ans ] = 0

            # 'concat' not necessary to have answer_vocab dimension
            elif self.config.output_combine == "concat":
                # Output order should be alphabetical
                assert self.config.graph_module.output_order == "alpha"


        # if model uses prior based on answer vocabulary
        if self.config.classifier.prior:

            # images
            unprocessed_priors = load_priors(self.config.classifier.prior_path)

            # priors have same size as answer_vocab
            assert len(img_priors) == self.config.classifier.params.out_dim
            # classifier is sigmoid (binary)
            assert 'sigmoid' == self.config.classifier.type

            # TODO: assert sigmoid?

            # list of image priors per answer
            self.priors = torch.tensor([])
            # iterate through each answer in answer_vocab
            for ans, ans_images in unprocessed_priors.items():

                # generating text priors
                text_features = self.text_processor({'text': ans})
                ans_text_prior = torch.flatten(text_features, start_dim=1)

                # generating image priors
                # get features from image priors
                image_features = self.image_processor({'image': ans_images})
                # average pool K features of size 2048
                # doing it on batches, and the grids e.g. 7x7 to get dim 2048
                ans_image_prior = torch.mean(image_features, dim=(0, 2, 3))
                ans_image_prior = torch.flatten(ans_image_prior, start_dim=1)

                combined = torch.cat([ans_text_prior, ans_image_prior], dim=0)
                # append row-wise
                self.priors = torch.cat([self.priors, combined.unsqueeze(0)])
                #priors.append(tuple(ans_image_prior, ans_text_prior))


    def forward(self, sample_list):

        # question encoding
        # Text input features will be in "input_ids" key
        question = sample_list["input_ids"]
        # Get the text and image features from the encoders
        question_features = self.language_module(question)# TODO: [1] in bert encoder?
        question_features = torch.flatten(question_features, start_dim=1)


        # image encoding
        image = sample_list["image"]
        image_features = self.vision_module(image)
        #print('img: ', image_features.shape)
        # TODO: average pooling, lots of other options (top-down, sum, multi)
        #   - text-embedding and _operator has good example
        if self.config.image_encoder.resize == 'average_pooling':
            # average pool K features of size 2048
            # doing it on dimensions 2 and 3 and keep 2048
            image_features = torch.mean(image_features, dim = (2,3))
        elif self.config.image_encoder.resize == 'none':
            # image feature dim is only 2048
            # assert self.config.image_encoder.type in ["resnet50", ...] # TODO: ?
            pass

        # Flatten the embeddings before concatenation
        image_features = torch.flatten(image_features, start_dim=1)


        # if using external knowledge (graph)
        if self.config.graph_module.use:
            # initialize for graph module
            sample_list["q_encoded"] = text # dim 128
            # Forward through graph module
            graph_output = self.graph_module(sample_list) # [128, 1310, 128]

            # logits from the  the output of the network
            if self.config.graph_module.graph_logit_mode == "in_graph":
                # Logits is already computed
                graph_logits = graph_output

            elif self.config.graph_module.graph_logit_mode == "logit_fc":
                # Compute logits from single hidden layer
                graph_logits = self.graph_logit_fc(graph_output)


            # combining features
            if self.config.graph_module.output_combine == "concat":
                # Combine both logits
                #logits = torch.cat([vb_logits, graph_logits], dim=1)
                #print('graph', graph_logits.shape)

                fusion = torch.cat([question_features, image_features, graph_logits], dim=1)
                # TODO: ?

            elif self.config.graph_module.output_combine == "add":
                # Set invalid inds to zero here
                assert graph_logits.size(1) == self.config.num_labels
                graph_logits[:, self.missing_ans_inds] = 0
                # TODO: ?

            # Now combine hidden dims
            #graph_output = torch.mean(graph_output, dim = 2) # mean pooling hidden dim of 768 so now batch*1310

            # Do zerobias
            # TODO: where does this come from?
            #if self.config.zerobias:
            #    logits -= 6.58


        # GatedTanh(in_dim, out_dim)


        # classifying
        if self.config.classifier.prior and self.config.classifier.type == 'sigmoid':

            # concatinating features
            combined = torch.cat([question_features, image_features], dim=1)
            # multiplying features on priors per answer/candidate in vocab
            combind_with_priors = torch.mul(self.priors, combined)
            # predictions scores for each candidate answer in vocab
            logits = torch.sigmoid(combind_with_priors)

        # mlp
        else:
            # Concatenate final features
            fused = torch.cat([question_features, image_features], dim=1)
            logits = self.classifier(fused)


        output = {"scores": logits}
        # MMF will automatically calculate loss
        return output







