
import sys

sys.path.insert(0, '/Users/arond.jacobsen/Documents/GitHub/explainableVQA/mmf')


from omegaconf import OmegaConf
import os
import numpy as np
import re
import matplotlib as mpl

from PIL import Image
import torch
import torchvision.transforms as T
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from mmf.utils.model_utils.image import openImage
from detectron2.modeling import postprocessing

from mmf.utils.build import (
    build_image_encoder,
    build_processors,
    build_detection_test_loader_for_images,
)
import tqdm
from mmf.utils.roi_heads import *
import cv2

transform = transforms.Compose([transforms.PILToTensor()])



if __name__ == "__main__":
    # getting configs
    config_path = 'mmexp/methods/bounding_boxes/dc5_50_not_norm.yaml'
    config = OmegaConf.load(os.path.dirname(os.getcwd()) + "/" + config_path)

    # loading images
    #general_path = "/imgs/temp/giraffe"
    general_path = "/imgs/temp/"

    image_file_names = ["oktennis"]  # , "sheep.jpg"]

    # building image processor
    processors = build_processors(config.dataset_config.okvqa.processors)
    image_processor = processors['image_processor']

    # loading image
    #original_images = []
    height, width = (224, 224)
    processed_images = []
    for file_name in image_file_names:
        # path to image
        image_path = os.path.dirname(os.getcwd()) + general_path + f"{file_name}/{file_name}.jpg"

        '''
        # the original image
        img = cv2.imread(os.path.dirname(image_path)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        # cropping img to 224, 224
        img = img[16:240, 16:240]
        #original_images.append(img[:, :, ::-1])
        '''

        # using the processor
        processed_image = image_processor({'image': openImage(image_path)})
        # loads the desired BGR
        # img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1)) # correct shape
        processed_images.append([processed_image])



    image_encoder = config.get("image_encoder", False)
    # building image encoder if defined
    if image_encoder:
        vision_module = build_image_encoder(image_encoder)
        vision_module.grid_feats_vqa.eval()
        #image_path = os.path.dirname(os.getcwd()) + general_path
        #data_loader = build_detection_test_loader_for_images(vision_module.cfg, image_path+image_files[0])

        #for idx, inputs in enumerate(tqdm.tqdm(data_loader)):
        for idx, inputs in enumerate(processed_images):
            #inputs = [{"image": image}]
            images = vision_module.grid_feats_vqa.preprocess_image(inputs)
            features = vision_module.grid_feats_vqa.backbone(images.tensor)
            proposals, _ = vision_module.grid_feats_vqa.proposal_generator(images, features)
            # pooled features and box predictions
            box_features, pooled_features_fc7, pooled_features_fc6 = vision_module.grid_feats_vqa.roi_heads.get_roi_features(
                features, proposals)
            print('the pooled', pooled_features_fc7.shape)
            predictions = vision_module.grid_feats_vqa.roi_heads.box_predictor(pooled_features_fc7)
            # predictions, r_indices = vision_module.grid_feats_vqa.roi_heads.box_predictor(box_features)
            predictions, r_indices = vision_module.grid_feats_vqa.roi_heads.box_predictor.inference(predictions, proposals)
            # print(r_indices[0].shape)
            box_type = type(proposals[0].proposal_boxes)
            proposal_bboxes = box_type.cat([p.proposal_boxes for p in proposals])
            proposal_bboxes.tensor = proposal_bboxes.tensor[r_indices]
            predictions[0].set("proposal_boxes", proposal_bboxes)
            r = postprocessing.detector_postprocess(predictions[0], height, width)
            bboxes = r.get("proposal_boxes").tensor
            print('after processing', bboxes.shape)

            # use the full model, this produces similar results
            #outputs = vision_module.grid_feats_vqa(inputs)
            #print('from outputs:', outputs[0]["instances"])


    for idx, img in enumerate(processed_images):
    #for idx, inputs in enumerate(tqdm.tqdm(data_loader)):
        img = processed_images[0][idx]['image'].permute(1, 2, 0) # correct shape
        #img = inputs[0]['image']
        img = img[:,:,[2, 1, 0]] # BGR to RGB
        # splitting non-characters and finding config file name
        exp_name = re.split('\W+', config_path)[-2]
        # image name
        v = Visualizer(img, None, scale=1.5)
        if image_encoder:
            #v = v.overlay_instances(boxes=outputs[0]["instances"].pred_boxes) # simply the same?
            v = v.overlay_instances(boxes=bboxes) # originally by grid-feats
            plot_img = v.get_image()
        else:
            plot_img = np.asarray(img).clip(0, 255).astype(np.uint8)

        # general plot configurations
        fig = plt.imshow(plot_img)
        mpl.rcParams['figure.dpi'] = 300
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(f'{os.path.dirname(os.getcwd())}/mmexp/methods/bounding_boxes/imgs/{exp_name}_{image_file_names[idx]}.jpg', bbox_inches='tight')
        plt.show(bbox_inches='tight')




