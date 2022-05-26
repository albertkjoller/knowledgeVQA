
import sys
from omegaconf import OmegaConf
import os
from PIL import Image
import torch
import torchvision.transforms as T
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import torchvision.transforms as transforms



from mmf.modules.encoders import gfvqaImageEncoder

from detectron2.checkpoint import DetectionCheckpointer

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
    
    model = gfvqaImageEncoder(config.image_encoder.params)
    
    model.cfg.defrost()
    #model.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.95
    #model.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.95
    model.cfg.freeze()
        
    #model = build_model(cfg)
    #DetectionCheckpointer(model.grid_feats_vqa).resume_or_load(model.cfg.MODEL.WEIGHTS, resume=True);
    
    path_to_img = "/imgs/protocol/COCO_val2014_000000000136.jpg"
    
    #path_to_coco_img = "/imgs/COCO_val2014_000000000136.jpg"
    img = Image.open(os.path.dirname(os.getcwd()) + path_to_img).convert('RGB')
    model.grid_feats_vqa.eval()
    
    
    #img_tensor = transform(img)
    
    #inputs = [{"image": img_tensor}]
    
    #images = model.grid_feats_vqa.preprocess_image(inputs) # Normalize, pad and batch the input images.
    #features = model.grid_feats_vqa.backbone(images.tensor) # features from backbone
    #outputs = model.grid_feats_vqa.roi_heads.get_conv5_features(features)
    inputs = [{"image": T.functional.to_tensor(img)}]
    with torch.no_grad():
        images = model.grid_feats_vqa.preprocess_image(inputs)
        features = model.grid_feats_vqa.backbone(images.tensor)
        proposals, _ = model.grid_feats_vqa.proposal_generator(images, features)
        # pooled features and box predictions
        box_features, pooled_features_fc7, pooled_features_fc6 = model.grid_feats_vqa.roi_heads.get_roi_features(
            features, proposals)
        predictions = model.grid_feats_vqa.roi_heads.box_predictor(pooled_features_fc7)
        predictions, r_indices = model.grid_feats_vqa.roi_heads.box_predictor.inference(predictions, proposals)
        print(predictions)
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




