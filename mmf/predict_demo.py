import os, gc
from pathlib import Path
from argparse import Namespace

import torch
import torch.nn.functional as F

from mmf.common.registry import registry
from mmf.common.sample import Sample, SampleList
from mmf.utils.env import setup_imports
from mmf.utils.configuration import Configuration
from mmf.models import *
from mmf.utils.build import build_processors

from utils.config import loadConfig
from utils.image import openImage
from utils.modeling import _multi_gpu_state_to_single

setup_imports()

class PretrainedModel:
    global ROOT_DIR
    ROOT_DIR = os.getcwd()

    def __init__(self, model_filename: str, ModelClass: type(BaseModel), dataset: str):
        self.model_filename = model_filename
        self.model_name = ('_').join(self.model_filename.split('_')[:-1])  # model is saved as "model_name_final.pth"
        self.ModelClass = ModelClass
        self.dataset = dataset

        self._init_processors()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vqa_model = self._build_vqa_model()

    def _init_processors(self):
        # define arguments
        args = Namespace()
        args.opts = [
            f"config={Path(f'{ROOT_DIR}/mmf/save/models/first_model/config.yaml')}",
            f"datasets={self.dataset}",
            f"model={self.model_name}",
            "evaluation.predict=True",
        ]
        args.config_override = None

        # read configurations
        configuration = Configuration(args=args)
        config = self.config = configuration.config
        dataset_config = config.dataset_config[self.dataset]

        # update .cache paths (different from the computer on which model was trained)
        cache_dir = str(Path.home() / '.cache/torch/mmf')
        data_dir = str(Path(cache_dir + '/data/datasets'))
        config.env.cache_dir, config.env.data_dir, dataset_config.data_dir = cache_dir, \
                                                                             str(Path(cache_dir + '/data')), \
                                                                             data_dir

        # update filepaths for dataset configuration processors
        dcp = dataset_config.processors
        dcp.text_processor.params.vocab.vocab_file = str(Path(f'{data_dir}/{dcp.text_processor.params.vocab.vocab_file}'))
        dcp.answer_processor.params.vocab_file = str(Path(f'{data_dir}/{dcp.answer_processor.params.vocab_file}'))

        # add processors
        processors = self.processors = build_processors(dataset_config.processors)
        self.text_processor = processors['text_processor']
        self.answer_processor = processors['answer_processor']
        self.image_processor = processors['image_processor']

        # TODO: check if we have to do this - I don't think it is necessary
        # register processors and
        #registry.register(f"{self.model_name}_text_processor", self.text_processor)
        #registry.register(f"{self.model_name}_answer_processor", self.answer_processor)
        #registry.register(f"{self.model_name}_num_final_outputs",
        #                  self.answer_processor.get_vocab_size())

    def _build_vqa_model(self):
        # load configuration and create model object
        config = loadConfig(self.model_filename)
        model = self.ModelClass(config)

        # specify path to saved model
        model_path = Path(f"{ROOT_DIR}/mmf/save/models/{self.model_name}/{self.model_filename}.pth")

        # load state dict and eventually convert from multi-gpu to single
        state_dict = torch.load(model_path)
        if list(state_dict.keys())[0].startswith('module') and not hasattr(model, 'module'):
            state_dict = _multi_gpu_state_to_single(state_dict)

        # load state dict to model
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_path, question, topk=5):
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
            topK = [(score.item(), self.answer_processor.idx2word(indices[0][idx].item())) for (idx, score) in enumerate(scores[0])]
            probs, answers = list(zip(*topK))

        # clean - garbage collection :TODO: why is this?
        gc.collect()
        torch.cuda.empty_cache()

        return probs, answers

if __name__ == '__main__':

    # specify model arguments and load model
    kwargs = {'model_filename': 'first_model_final',
            'ModelClass': First_Model,
            'dataset': 'okvqa',
            }
    FirstModel = PretrainedModel(**kwargs)

    # input for prediction
    img_path = r'C:\Users\Bruger\Desktop\rain.jpg'
    question = 'How is the weather?'
    topk = 5

    # get predictions
    outputs = FirstModel.predict(image_path=img_path, question=question, topk=topk)
    print("\nPredicted outputs from the model:")
    for i, (prob, answer) in enumerate(zip(*outputs)):
        print(f"{i+1}) {answer} \t ({prob})")

