import os, sys

# project-directory
#PROJECT_DIR = os.getcwd()

# add pytorch_vqa and pytorch_resnet to filepath
#sys.path.append(os.path.realpath(f"{PROJECT_DIR}/pytorch-vqa"))
#sys.path.append(os.path.realpath(f"{PROJECT_DIR}/pytorch-resnet"))

import threading
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from pytorch_resnet import resnet  # from pytorch_resnet

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

from pytorch_vqa.model import Net, apply_attention, tile_2d_over_nd # from pytorch_vqa
from pytorch_vqa.utils import get_transform # from pytorch_vqa

from captum.attr import (
    IntegratedGradients,
    LayerIntegratedGradients,
    TokenReferenceBase,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
    visualization
)
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper


# demo taken from here: https://captum.ai/tutorials/Multimodal_VQA_Interpret
# and from here:        https://captum.ai/tutorials/Multimodal_VQA_Captum_Insights

##### ----- Start ----- #####

# setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load
saved_state = torch.load('captum_vqa/models/2017-08-04_00.55.19.pth', map_location=device)

# reading vocabulary, tokens and answers from saved model
vocab = saved_state['vocab']
token_to_index = vocab['question']
answer_to_index = vocab['answer']

# compute number of tokens
num_tokens = len(token_to_index) + 1

# reading answer classes from the vocabulary
answer_words = ['unk'] * len(answer_to_index)
for w, idx in answer_to_index.items():
    answer_words[idx] = w

# load predefined model
vqa_net = torch.nn.DataParallel(Net(num_tokens))
vqa_net.load_state_dict(saved_state['weights'])
vqa_net.to(device)
vqa_net.eval() # sets the network/module in evaluation mode - #TODO: can be changed with vqa_net.train(True)


# from https://github.com/Cyanogenoid/pytorch-vqa/blob/master/data.py#L110
def encode_question(question):
    """ Turn a question into a vector of indices and a question length """
    question_arr = question.lower().split()
    vec = torch.zeros(len(question_arr), device=device).long()
    for i, token in enumerate(question_arr):
        index = token_to_index.get(token, 0)
        vec[i] = index
    return vec, torch.tensor(len(question_arr), device=device)


class ResNetLayer4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.r_model = resnet.resnet152(pretrained=True)
        self.r_model.eval()
        self.r_model.to(device)

        self.buffer = {}
        lock = threading.Lock() #TODO: understand this

        # Since we only use the output of the 4th layer from the resnet model and do not
        # need to do forward pass all the way to the final layer we can terminate forward
        # execution in the forward hook of that layer after obtaining the output of it.
        # For that reason, we can define a custom Exception class that will be used for
        # raising early termination error.
        def save_output(module, input, output):
            with lock: #TODO: related to threading above
                self.buffer[output.device] = output

        self.r_model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.r_model(x)
        return self.buffer[x.device]


class VQA_Resnet_Model(Net):
    def __init__(self, embedding_tokens):
        super().__init__(embedding_tokens)
        self.resnet_layer4 = ResNetLayer4()

    def forward(self, v, q, q_len):
        q = self.text(q, list(q_len.data))
        v = self.resnet_layer4(v)

        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)

        a = self.attention(v, q)
        v = apply_attention(v, a)

        combined = torch.cat([v, q], dim=1)
        answer = self.classifier(combined)
        return answer


# below: interpretabel something with their integrated gradients TODO: find out what their good idea is and how it works

# Option 1): "Patch" the model's embedding layer and corresponding inputs. To patch the layer, use the
# configure_interpretable_embedding_layer^ method, which will wrap the associated layer you give it, with an identity
# function. This identity function accepts an embedding and outputs an embedding. You can patch the inputs, i.e. obtain
# the embedding for a set of indices, with model.wrapped_layer.indices_to_embeddings(indices).

# Option 2): Use the equivalent layer attribution algorithm (LayerIntegratedGradients in our case) with the utility
# class ModelInputWrapper. The ModelInputWrapper will wrap your model and feed all it's inputs to seperate layers;
# allowing you to use layer attribution methods on inputs. You can access the associated layer for input named "foo"
# via the ModuleDict: wrapped_model.input_maps["foo"].

USE_INTERPRETABLE_EMBEDDING_LAYER = False  # set to True for option (1)


# ----- update weights from saved model and remove old model from memory -----
vqa_resnet = VQA_Resnet_Model(vqa_net.module.text.embedding.num_embeddings)
vqa_resnet = ModelInputWrapper(vqa_resnet) # wrap the inputs into layers incase we wish to use a layer method
vqa_resnet = torch.nn.DataParallel(vqa_resnet) # paralelization with `device_ids` supported by `DataParallel`

# saved vqa model's parameters #TODO: find out what happens here
partial_dict = vqa_net.state_dict()

state = vqa_resnet.module.state_dict()
state.update(partial_dict)
vqa_resnet.module.load_state_dict(state)

vqa_resnet.to(device)
vqa_resnet.eval()

del vqa_net # This is original VQA model without resnet. Removing it, since we do not need it


# patch models embedding layer if option 1) is used
if USE_INTERPRETABLE_EMBEDDING_LAYER:
    interpretable_embedding = configure_interpretable_embedding_layer(vqa_resnet, 'module.module.text.embedding')




image_size = 448  # scale image to given size and center
central_fraction = 1.0

transform = get_transform(image_size, central_fraction=central_fraction)


def image_to_features(img):
    img_transformed = transform(img)
    img_batch = img_transformed.unsqueeze(0).to(device)
    return img_batch


PAD_IND = token_to_index['pad']
token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)

# this is necessary for the backpropagation of RNNs models in eval mode
torch.backends.cudnn.enabled=False


if USE_INTERPRETABLE_EMBEDDING_LAYER:
    attr = IntegratedGradients(vqa_resnet)
else:
    attr = LayerIntegratedGradients(vqa_resnet, [vqa_resnet.module.input_maps["v"], vqa_resnet.module.module.text.embedding])

default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.25, '#252b36'),
                                                  (1, '#000000')], N=256)


def vqa_resnet_interpret(image_filename, questions, targets):
    img = Image.open(image_filename).convert('RGB')
    original_image = transforms.Compose([transforms.Resize(int(image_size / central_fraction)),
                                         transforms.CenterCrop(image_size), transforms.ToTensor()])(img)

    image_features = image_to_features(img).requires_grad_().to(device)
    for question, target in zip(questions, targets):
        q, q_len = encode_question(question)

        # generate reference for each sample
        q_reference_indices = token_reference.generate_reference(q_len.item(), device=device).unsqueeze(0)

        inputs = (q.unsqueeze(0), q_len.unsqueeze(0))
        if USE_INTERPRETABLE_EMBEDDING_LAYER:
            q_input_embedding = interpretable_embedding.indices_to_embeddings(q).unsqueeze(0)
            q_reference_baseline = interpretable_embedding.indices_to_embeddings(q_reference_indices).to(device)

            inputs = (image_features, q_input_embedding)
            baselines = (image_features * 0.0, q_reference_baseline)

        else:
            inputs = (image_features, q.unsqueeze(0))
            baselines = (image_features * 0.0, q_reference_indices)

        ans = vqa_resnet(*inputs, q_len.unsqueeze(0))

        # Make a prediction. The output of this prediction will be visualized later.
        pred, answer_idx = F.softmax(ans, dim=1).data.cpu().max(dim=1)

        attributions = attr.attribute(inputs=inputs,
                                      baselines=baselines,
                                      target=answer_idx,
                                      additional_forward_args=q_len.unsqueeze(0),
                                      n_steps=30)

        # Visualize text attributions
        text_attributions_norm = attributions[1].sum(dim=2).squeeze(0).norm()
        vis_data_records = [visualization.VisualizationDataRecord(
            attributions[1].sum(dim=2).squeeze(0) / text_attributions_norm,
            pred[0].item(),
            answer_words[answer_idx],
            answer_words[answer_idx],
            target,
            attributions[1].sum(),
            question.split(),
            0.0)]
        visualization.visualize_text(vis_data_records)

        # visualize image attributions
        original_im_mat = np.transpose(original_image.cpu().detach().numpy(), (1, 2, 0))
        attributions_img = np.transpose(attributions[0].squeeze(0).cpu().detach().numpy(), (1, 2, 0))

        visualization.visualize_image_attr_multiple(attributions_img, original_im_mat,
                                                    ["original_image", "heat_map"], ["all", "absolute_value"],
                                                    titles=["Original Image", "Attribution Magnitude"],
                                                    cmap=default_cmap,
                                                    show_colorbar=True)
        print('Text Contributions: ', attributions[1].sum().item())
        print('Image Contributions: ', attributions[0].sum().item())
        print('Total Contribution: ', attributions[0].sum().item() + attributions[1].sum().item())

images = ['./captum_vqa/img/vqa/siamese.jpg',
          './captum_vqa/img/vqa/elephant.jpg',
          './captum_vqa/img/vqa/zebra.jpg']

import IPython
# the index of image in the test set. Please, change it if you want to play with different test images/samples.
image_idx = 1 # elephant
vqa_resnet_interpret(images[image_idx],
                     ["what is on the picture",
                      "what color is the elephant",
                      "where is the elephant"],
                     ['elephant', 'gray', 'zoo'])
IPython.display.Image(filename='captum_vqa/img/vqa/elephant_attribution.jpg')


image_idx = 0 # cat
vqa_resnet_interpret(images[image_idx], [
    "what is on the picture",
    "what color are the cat's eyes",
    "is the animal in the picture a cat or a fox",
    "what color is the cat",
    "how many ears does the cat have",
    "where is the cat"
], ['cat', 'blue', 'cat', 'white and brown', '2', 'at the wall'])
IPython.display.Image(filename='captum_vqa/img/vqa/siamese_attribution.jpg')

IPython.display.Image(filename='captum_vqa/img/vqa/elephant_attribution.jpg')



if USE_INTERPRETABLE_EMBEDDING_LAYER:
    remove_interpretable_embedding_layer(vqa_resnet, interpretable_embedding)

print("breakpoint")