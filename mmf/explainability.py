import os, gc
import pickle
from pathlib import Path
from argparse import Namespace

import torch
import torch.nn.functional as F

from torchray.attribution.guided_backprop import GuidedBackpropContext
from torchray.attribution.common import gradient_to_saliency


"""
from mmf.common.registry import registry
from mmf.common.sample import Sample, SampleList
from mmf.utils.env import setup_imports
from mmf.utils.configuration import Configuration
from mmf.models import *
from mmf.utils.build import build_processors

from mmf.utils.model_utils.config import loadConfig
from mmf.utils.model_utils.image import openImage
from mmf.utils.model_utils.modeling import _multi_gpu_state_to_single

setup_imports()
"""

class PretrainedModel:
    global ROOT_DIR
    ROOT_DIR = os.getcwd()

    def __init__(self, experiment_name: str, model_filename: str, ModelClass: type(BaseModel), dataset: str, GBAR=bool, studynumber=None):

        self.experiment_name = experiment_name
        self.model_filename = model_filename
        self.model_name = ('_').join(self.model_filename.split('_')[:-1])  # model is saved as "model_name_final.pth"
        self.ModelClass = ModelClass
        self.dataset = dataset
        self.GBAR = GBAR
        self.studynumber = studynumber

        self._init_processors()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vqa_model = self._build_vqa_model()

    def _init_processors(self):
        # define arguments
        args = Namespace()
        if self.GBAR:
            config_path = Path(f'/work3/{self.studynumber}/Bachelor/save/models/{self.experiment_name}/config.yaml')
        else:
            config_path = Path(f'{ROOT_DIR}/mmf/save/models/{self.experiment_name}/config.yaml')
            config_path = Path(f'{ROOT_DIR}/save/models/{self.experiment_name}/config.yaml')

        args.opts = [
            f"config={config_path}",
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

        if self.GBAR:
            data_dir = '/work3/{:s}'.format(self.studynumber)
        else:
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

    def _build_vqa_model(self):
        # load configuration and create model object
        if self.GBAR:
            model_path = Path(f"/work3/{self.studynumber}/Bachelor/save/models/{self.experiment_name}/{self.model_filename}.pth")
            config = loadConfig(self.experiment_name, self.model_name, studynumber=studynumber)

        else:
            model_path = Path(f"{ROOT_DIR}/save/models/{self.experiment_name}/{self.model_filename}.pth")
            config = loadConfig(self.experiment_name, self.model_name)
        model = self.ModelClass(config)

        # load state dict and eventually convert from multi-gpu to single
        if self.device == 'cpu':
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        else:
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
            sample.tokens = processed_text['tokens']

            # process image input
            processed_image = self.image_processor({'image': openImage(image_path)})
            sample.image = processed_image['image']

            # gather in sample list
            sample_list = SampleList([sample]).to(self.device)

            sample_list['image'].requires_grad_(True)
            #sample_list['input_ids'].requires_grad_(True)

            # predict scores with model (multiclass)
            #scores = self.vqa_model(sample_list)["scores"]
            #scores = torch.nn.functional.softmax(scores, dim=1)
            
            with GuidedBackpropContext():
                  y = self.vqa_model(sample_list)['scores']
                  z = y[0, y.argmax()]
                  z.backward()
              
            saliency = gradient_to_saliency(sample_list['image'])
            

            # extract probabilities and answers for top k predicted answers
            scores, indices = scores.topk(topk, dim=1)
            topK = [(score.item(), self.answer_processor.idx2word(indices[0][idx].item())) for (idx, score) in enumerate(scores[0])]

            probs, answers = list(zip(*topK))

        # clean - garbage collection :TODO: why is this?
        gc.collect()
        torch.cuda.empty_cache()

        return probs, answers

if __name__ == '__main__':
    import sys, cv2

    # helper function for input
    def str_to_class(classname):
        return getattr(sys.modules['mmf.models'], classname)
    
    experiment_name = 'logit_mul'
    model_filename = 'qlarifais_final'
    ModelClass = 'Qlarifais'
    dataset = 'okvqa'
    GBAR = 'yes'
    studynumber = 's194253'
    
    experiment_name = 'first_model'
    model_filename = 'first_model_final'
    ModelClass = 'First_Model'
    dataset = 'okvqa'
    GBAR = 'yes'
    studynumber = 's194253'
    
    # specify model arguments and load model
    kwargs = {'experiment_name': experiment_name,
              'model_filename': model_filename,
              'ModelClass': str_to_class(ModelClass),
              'dataset': dataset,
              'GBAR': True if GBAR == 'yes' else False,
              'studynumber': studynumber if GBAR == 'yes' else None}

    model = PretrainedModel(**kwargs)

    # only when debugging - from bash this doesn't matter #TODO: remove in the end
    if os.getcwd().split(os.sep)[-1] != 'explainableVQA':
        os.chdir(os.path.dirname(os.getcwd()))
        
    img_name, question = 'rain.jpg', 'where is it raining'

    img_path = Path(f"{os.getcwd()}/imgs/temp/{img_name}").as_posix()
    img = cv2.imread(img_path)  # open from file object

    # calculate the 50 percent of original dimensions
    width = int(img.shape[1] * 0.2)
    height = int(img.shape[0] * 0.2)

    # get predictions and show input
    topk = 5
    outputs = model.predict(image_path=img_path, question=question, topk=topk)

    # print answers and probabilities
    print(f'\nQuestion: "{question}"')
    print("\nPredicted outputs from the model:")
    for i, (prob, answer) in enumerate(zip(*outputs)):
        print(f"{i+1}) {answer} \t ({prob})")
