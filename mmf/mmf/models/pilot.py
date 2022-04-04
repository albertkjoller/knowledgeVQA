
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


'''
fast run:
mmf_run config='configs/experiments/pilot/grids.yaml' \
    datasets=okvqa \
    model=pilot \
    run_type=train_val \
    training.batch_size=32 \
    training.max_updates=1 \

'''

# Register the model for MMF, "concat_bert_tutorial" key would be used to find the model
@registry.register_model("pilot")
class Pilot(BaseModel):
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



    def forward(self, sample_list):
        # Each model in MMF gets a dict called sample_list which contains
        # all of the necessary information returned from the image

        # Text input features will be in "input_ids" key
        text = sample_list["input_ids"]
        # Similarly, image input will be in "image" key
        image = sample_list["image"]

        # Get the text and image features from the encoders
        text_features = self.language_module(text) #TODO: with [1] if bert uncased
        image_features = self.vision_module(image)

        # TODO: average pooling, lots of other options (top-down, sum, multi)
        #   - text-embedding and _operator has good example
        # grids: [batch_size, 2048, 7, 7] (7*7=49 grids)
        #   - img: [224, 224] in which (224*224)/1024 = 49
        image_features = torch.mean(image_features, dim = (2,3))

        # Flatten the embeddings before concatenation
        image_features = torch.flatten(image_features, start_dim=1)
        text_features = torch.flatten(text_features, start_dim=1)

        # Concatenate final features
        combined = torch.cat([text_features, image_features], dim=1) # TODO: (top down bottom up) haardman product

        # Pass final tensor to classifier to get scores
        logits = self.classifier(combined)
        output = {"scores": logits}

        # MMF will automatically calculate loss
        return output

'''
    def load(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # load experiment config
        self.loadExperimentConfig()
        model = self(self.experiment_config)

        # specify path to saved model
        ROOT_DIR = os.getcwd()
        model_path = Path(f"{ROOT_DIR}/mmf/save/models/{model_name}/{model_name}_final.pth")

        # load state dict and convert from multi-gpu to single (in case)
        state_dict = torch.load(model_path)
        if list(state_dict.keys())[0].startswith('module') and not hasattr(model, 'module'):
            state_dict = self._multi_gpu_state_to_single(state_dict)

        # load state dict to model
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def loadExperimentConfig(self):
        # load experiment-configuration and create model object
        experiment_config = self.experiment_config = loadConfig(self.model_name)
        dataset_config = self.dataset_config = experiment_config.dataset_config

        # update .cache paths (different from the computer on which model was trained)
        cache_dir = str(Path.home() / '.cache/torch/mmf')
        data_dir = str(Path(cache_dir + '/data/datasets'))
        experiment_config.env.cache_dir, experiment_config.env.data_dir, dataset_config.data_dir = cache_dir, \
                                                                                                   str(Path(
                                                                                                       cache_dir + '/data')), \
                                                                                                   data_dir
        # update filepaths for dataset configuration processors
        dcp = dataset_config.processors
        dcp.text_processor.params.vocab.vocab_file = str(Path(f'{data_dir}/{dcp.text_processor.params.vocab.vocab_file}'))
        dcp.answer_processor.params.vocab_file = str(Path(f'{data_dir}/{dcp.answer_processor.params.vocab_file}'))

    def predict(self, image_path, question, topk=5):

        # build processors
        processors = self.processors = build_processors(self.dataset_config[dataset_name])
        self.text_processor = processors['text_processor']
        self.answer_processor = processors['answer_processor']
        self.image_processor = processors['image_processor']

        with torch.no_grad():
            # create sample object - required for model input
            sample = Sample()

            # process text input
            processed_text = self.text_processor({'text': question})
            sample.input_ids = processed_text['input_ids']
            sample.text_len = len(processed_text['tokens'])

            # process image input
            processed_image = self.image_processor({'image': openImage(image_path)})
            sample.image = processed_image['image']

            # gather in sample list
            sample_list = SampleList([sample]).to(self.device)

            # predict scores with model (multiclass)
            scores = self.vqa_model(sample_list)["scores"]
            scores = torch.nn.functional.softmax(scores, dim=1)

            # extract probabilities and answers for top k predicted answers
            scores, indices = scores.topk(topk, dim=1)
            topK = [(score.item(), self.answer_processor.idx2word(indices[0][idx].item())) for (idx, score) in
                    enumerate(scores[0])]
            probs, answers = list(zip(*topK))

        # clean - garbage collection :TODO: why is this?
        gc.collect()
        torch.cuda.empty_cache()

        return probs, answers

'''






