import os
from pathlib import Path
from argparse import Namespace

import torch, torchvision

from utils.config import loadConfig
from mmf.models import *


model_name = 'first_model_final'
config = loadConfig(model_name)

model = First_Model(config)
loadedModelPath = Path(f"{os.path.dirname(os.getcwd())}/mmf/save/models/{('_').join(model_name.split('_')[:-1])}/{model_name}.pth")
model.load_state_dict(torch.load(loadedModelPath))

class MMFDemo:
    TARGET_IMAGE_SIZE = [448, 448]
    CHANNEL_MEAN = [0.485, 0.456, 0.406]
    CHANNEL_STD = [0.229, 0.224, 0.225]

    def __init__(self):
        self._init_processors()
        self.model = self._build_pythia_model()
        self.resnet_model = self._build_resnet_model()

    def _init_processors(self):
        args = Namespace()
        args.opts = [
            "config=projects/pythia/configs/vqa2/defaults.yaml",
            "datasets=vqa2",
            "model=pythia",
            "evaluation.predict=True"
        ]
        args.config_override = None

        configuration = Configuration(args=args)

        config = self.config = configuration.config
        vqa_config = config.dataset_config.vqa2
        text_processor_config = vqa_config.processors.text_processor
        answer_processor_config = vqa_config.processors.answer_processor

        text_processor_config.params.vocab.vocab_file = "/content/model_data/vocabulary_100k.txt"
        answer_processor_config.params.vocab_file = "/content/model_data/answers_vqa.txt"
        # Add preprocessor as that will needed when we are getting questions from user
        self.text_processor = VocabProcessor(text_processor_config.params)
        self.answer_processor = VQAAnswerProcessor(answer_processor_config.params)

        registry.register("vqa2_text_processor", self.text_processor)
        registry.register("vqa2_answer_processor", self.answer_processor)
        registry.register("vqa2_num_final_outputs",
                          self.answer_processor.get_vocab_size())

    def _build_pythia_model(self):
        state_dict = torch.load('/content/model_data/pythia.pth')
        model_config = self.config.model_config.pythia
        model_config.model_data_dir = "/content/"
        model = Pythia(model_config)
        model.build()
        model.init_losses()