
import sys
from omegaconf import OmegaConf
import os
from PIL import Image
import torch
import torchvision.transforms as T
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from detectron2.evaluation import inference_context
from detectron2.modeling import postprocessing

from mmf.utils.build import build_image_encoder

import cv2
transform = transforms.Compose([
transforms.PILToTensor()])

sys.path.append("..")

#from mmf.models.frcnn import GeneralizedRCNN
# python3 -u /examples/qlarifais_bounding_boxes.py

if __name__ == "__main__":
    
    config_path = 'mmexp/methods/bounding_boxes/config2.yaml'
    
    config = OmegaConf.load(os.path.dirname(os.getcwd()) + "/" + config_path)
    # Init FasterRCNN
    #model = GeneralizedRCNN(config)
    vision_module = build_image_encoder(config.image_encoder)
    
    vision_module.cfg.defrost()
    #vision_module.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.00001
    vision_module.cfg.freeze()

    #path_to_img = "/imgs/protocol/COCO_val2014_000000000136.jpg"
    path_to_img = "/imgs/protocol/sheep1.jpg"
    # loading image
    #img = Image.open(os.path.dirname(os.getcwd()) + path_to_img).convert('RGB')
    img = cv2.imread(os.path.dirname(os.getcwd()) + path_to_img)
    height, width = img.shape[:2]
    img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1)) # BGR
    inputs = [{"image": img}]#, "height": height, "width": width}]
    vision_module.grid_feats_vqa.eval()
    #inputs = [{"image": T.functional.to_tensor(img)}]
    #with inference_context(vision_module.grid_feats_vqa):
    #with torch.no_grad():
    images = vision_module.grid_feats_vqa.preprocess_image(inputs)
    features = vision_module.grid_feats_vqa.backbone(images.tensor)
    proposals, _ = vision_module.grid_feats_vqa.proposal_generator(images, features)
    # pooled features and box predictions
    box_features, pooled_features_fc7, pooled_features_fc6 = vision_module.grid_feats_vqa.roi_heads.get_roi_features(
        features, proposals)
    predictions = vision_module.grid_feats_vqa.roi_heads.box_predictor(pooled_features_fc7)
    #predictions, r_indices = vision_module.grid_feats_vqa.roi_heads.box_predictor(box_features)
    predictions, r_indices = vision_module.grid_feats_vqa.roi_heads.box_predictor.inference(predictions, proposals)
    print(r_indices)
    raise NotImplementedError
    box_type = type(proposals[0].proposal_boxes)
    proposal_bboxes = box_type.cat([p.proposal_boxes for p in proposals])
    proposal_bboxes.tensor = proposal_bboxes.tensor[r_indices]
    predictions[0].set("proposal_boxes", proposal_bboxes)
    # postprocess
    r = postprocessing.detector_postprocess(predictions[0], height, width)

    bboxes = r.get("proposal_boxes").tensor
    print(bboxes)
    raise NotImplementedError


    #outputs = model.grid_feats_vqa([{"image": T.functional.to_tensor(img)}])
    # (img[:, :, ::-1] # convert RGB to BGR
    v_gt = Visualizer(img, None)
    v_gt = v_gt.overlay_instances(boxes=outputs[0]["instances"].pred_boxes)
    print(outputs)
    #print(outputs["instances"].pred_boxes)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(v_gt.get_image())
    plt.show()




