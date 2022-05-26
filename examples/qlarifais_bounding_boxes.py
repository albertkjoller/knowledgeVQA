
import sys
from omegaconf import OmegaConf
import os
from PIL import Image
import torch
import torchvision.transforms as T
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from detectron2.modeling import postprocessing


from mmf.utils.build import build_image_encoder

from mmf.utils.roi_heads import *

import cv2
transform = transforms.Compose([
transforms.PILToTensor()])

sys.path.append("..")

#from mmf.models.frcnn import GeneralizedRCNN
# python3 -u /examples/qlarifais_bounding_boxes.py


class overwrite_detectron:
    def __init__(self):

        import inspect
        from typing import List, Optional
        import torch
        from torch import nn
        from torch.nn import functional as F

        from detectron2.config import configurable
        from detectron2.layers import ShapeSpec
        from detectron2.modeling.roi_heads import (
            build_box_head,
            build_mask_head,
            select_foreground_proposals,
            ROI_HEADS_REGISTRY,
            ROI_BOX_HEAD_REGISTRY,
            ROIHeads,
            Res5ROIHeads,
            StandardROIHeads,
        )

        from detectron2.modeling.roi_heads.box_head import FastRCNNConvFCHead
        from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
        from detectron2.modeling.poolers import ROIPooler

        @ROI_BOX_HEAD_REGISTRY.register()
        class AttributeFastRCNNConvFCHead(FastRCNNConvFCHead):
            """
            Modified version of FastRCNNConvFCHead which output last two FC outputs
            """

            def forward(self, x):
                for layer in self.conv_norm_relus:
                    x = layer(x)
                y = None
                if len(self.fcs):
                    if x.dim() > 2:
                        x = torch.flatten(x, start_dim=1)
                    for layer in self.fcs:
                        y = x
                        x = F.relu(layer(y))
                return x, y

        class AttributePredictor(nn.Module):
            """
            Head for attribute prediction, including feature/score computation and
            loss computation.

            """

            def __init__(self, cfg, input_dim):
                super().__init__()

                # fmt: off
                self.num_objs = cfg.MODEL.ROI_HEADS.NUM_CLASSES
                self.obj_embed_dim = cfg.MODEL.ROI_ATTRIBUTE_HEAD.OBJ_EMBED_DIM
                self.fc_dim = cfg.MODEL.ROI_ATTRIBUTE_HEAD.FC_DIM
                self.num_attributes = cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_CLASSES
                self.max_attr_per_ins = cfg.INPUT.MAX_ATTR_PER_INS
                self.loss_weight = cfg.MODEL.ROI_ATTRIBUTE_HEAD.LOSS_WEIGHT
                # fmt: on

                # object class embedding, including the background class
                self.obj_embed = nn.Embedding(self.num_objs + 1, self.obj_embed_dim)
                input_dim += self.obj_embed_dim
                self.fc = nn.Sequential(
                    nn.Linear(input_dim, self.fc_dim),
                    nn.ReLU()
                )
                self.attr_score = nn.Linear(self.fc_dim, self.num_attributes)
                nn.init.normal_(self.attr_score.weight, std=0.01)
                nn.init.constant_(self.attr_score.bias, 0)

            def forward(self, x, obj_labels):
                attr_feat = torch.cat((x, self.obj_embed(obj_labels)), dim=1)
                return self.attr_score(self.fc(attr_feat))

            def loss(self, score, label):
                n = score.shape[0]
                score = score.unsqueeze(1)
                score = score.expand(n, self.max_attr_per_ins, self.num_attributes).contiguous()
                score = score.view(-1, self.num_attributes)
                inv_weights = (
                    (label >= 0).sum(dim=1).repeat(self.max_attr_per_ins, 1).transpose(0, 1).flatten()
                )
                weights = inv_weights.float().reciprocal()
                weights[weights > 1] = 0.
                n_valid = len((label >= 0).sum(dim=1).nonzero())
                label = label.view(-1)
                attr_loss = F.cross_entropy(score, label, reduction="none", ignore_index=-1)
                attr_loss = (attr_loss * weights).view(n, -1).sum(dim=1)

                if n_valid > 0:
                    attr_loss = attr_loss.sum() * self.loss_weight / n_valid
                else:
                    attr_loss = attr_loss.sum() * 0.
                return {"loss_attr": attr_loss}

        class AttributeROIHeads(ROIHeads):
            """
            An extension of ROIHeads to include attribute prediction.
            """

            def forward_attribute_loss(self, proposals, box_features):
                proposals, fg_selection_attributes = select_foreground_proposals(
                    proposals, self.num_classes
                )
                attribute_features = box_features[torch.cat(fg_selection_attributes, dim=0)]
                obj_labels = torch.cat([p.gt_classes for p in proposals])
                attribute_labels = torch.cat([p.gt_attributes for p in proposals], dim=0)
                attribute_scores = self.attribute_predictor(attribute_features, obj_labels)
                return self.attribute_predictor.loss(attribute_scores, attribute_labels)

        @ROI_HEADS_REGISTRY.register()
        class AttributeRes5ROIHeads(AttributeROIHeads, Res5ROIHeads):
            """
            An extension of Res5ROIHeads to include attribute prediction.
            """

            def __init__(self, cfg, input_shape):
                super(Res5ROIHeads, self).__init__(cfg, input_shape)

                assert len(self.in_features) == 1

                # fmt: off
                pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
                pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
                pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)
                sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
                self.mask_on = cfg.MODEL.MASK_ON
                self.attribute_on = cfg.MODEL.ATTRIBUTE_ON
                # fmt: on
                assert not cfg.MODEL.KEYPOINT_ON

                self.pooler = ROIPooler(
                    output_size=pooler_resolution,
                    scales=pooler_scales,
                    sampling_ratio=sampling_ratio,
                    pooler_type=pooler_type,
                )

                self.res5, out_channels = self._build_res5_block(cfg)
                self.box_predictor = FastRCNNOutputLayers(
                    cfg, ShapeSpec(channels=out_channels, height=1, width=1)
                )

                if self.mask_on:
                    self.mask_head = build_mask_head(
                        cfg,
                        ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
                    )

                if self.attribute_on:
                    self.attribute_predictor = AttributePredictor(cfg, out_channels)

            def forward(self, images, features, proposals, targets=None):
                del images

                if self.training:
                    assert targets
                    proposals = self.label_and_sample_proposals(proposals, targets)
                del targets

                proposal_boxes = [x.proposal_boxes for x in proposals]
                box_features = self._shared_roi_transform(
                    [features[f] for f in self.in_features], proposal_boxes
                )
                feature_pooled = box_features.mean(dim=[2, 3])
                predictions = self.box_predictor(feature_pooled)

                if self.training:
                    del features
                    losses = self.box_predictor.losses(predictions, proposals)
                    if self.mask_on:
                        proposals, fg_selection_masks = select_foreground_proposals(
                            proposals, self.num_classes
                        )
                        mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                        del box_features
                        losses.update(self.mask_head(mask_features, proposals))
                    if self.attribute_on:
                        losses.update(self.forward_attribute_loss(proposals, feature_pooled))
                    return [], losses
                else:
                    pred_instances, _ = self.box_predictor.inference(predictions, proposals)
                    pred_instances = self.forward_with_given_boxes(features, pred_instances)
                    return pred_instances, {}

            def get_conv5_features(self, features):
                features = [features[f] for f in self.in_features]
                return self.res5(features[0])

            def get_roi_features(self, features, proposals):
                assert len(self.in_features) == 1

                features = [features[f] for f in self.in_features]
                box_features = self._shared_roi_transform(
                    features, [x.proposal_boxes for x in proposals]
                )
                pooled_features = box_features.mean(dim=[2, 3])
                return box_features, pooled_features, None

        @ROI_HEADS_REGISTRY.register()
        class AttributeStandardROIHeads(AttributeROIHeads, StandardROIHeads):
            """
            An extension of StandardROIHeads to include attribute prediction.
            """

            @configurable
            def __init__(
                    self,
                    *,
                    box_in_features: List[str],
                    box_pooler: ROIPooler,
                    box_head: nn.Module,
                    box_predictor: nn.Module,
                    mask_in_features: Optional[List[str]] = None,
                    mask_pooler: Optional[ROIPooler] = None,
                    mask_head: Optional[nn.Module] = None,
                    keypoint_in_features: Optional[List[str]] = None,
                    keypoint_pooler: Optional[ROIPooler] = None,
                    keypoint_head: Optional[nn.Module] = None,
                    train_on_pred_boxes: bool = False,
                    attribute_on: bool = False,
                    attribute_predictor: Optional[nn.Module] = None,
                    **kwargs
            ):
                super(StandardROIHeads, self).__init__(**kwargs)
                # keep self.in_features for backward compatibility
                self.in_features = self.box_in_features = box_in_features
                self.box_pooler = box_pooler
                self.box_head = box_head
                self.box_predictor = box_predictor

                self.mask_on = mask_in_features is not None
                if self.mask_on:
                    self.mask_in_features = mask_in_features
                    self.mask_pooler = mask_pooler
                    self.mask_head = mask_head
                self.keypoint_on = keypoint_in_features is not None
                if self.keypoint_on:
                    self.keypoint_in_features = keypoint_in_features
                    self.keypoint_pooler = keypoint_pooler
                    self.keypoint_head = keypoint_head

                self.train_on_pred_boxes = train_on_pred_boxes

                self.attribute_on = attribute_on
                if self.attribute_on:
                    self.attribute_predictor = attribute_predictor

            @classmethod
            def from_config(cls, cfg, input_shape):
                ret = super().from_config(cfg, input_shape)
                ret["attribute_on"] = cfg.MODEL.ATTRIBUTE_ON
                return ret

            @classmethod
            def _init_box_head(cls, cfg, input_shape):
                ret = StandardROIHeads._init_box_head(cfg, input_shape)
                if cfg.MODEL.ATTRIBUTE_ON:
                    ret.update({"attribute_predictor": AttributePredictor(cfg, ret["box_head"].output_shape.channels)})
                return ret

            def _forward_box(self, features, proposals):
                features = [features[f] for f in self.in_features]
                box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
                box_features, _ = self.box_head(box_features)
                predictions = self.box_predictor(box_features)

                if self.training:
                    if self.train_on_pred_boxes:
                        with torch.no_grad():
                            pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                                predictions, proposals
                            )
                            for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                                proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
                    losses = self.box_predictor.losses(predictions, proposals)
                    if self.attribute_on:
                        losses.update(self.forward_attribute_loss(proposals, box_features))
                        del box_features

                    return losses
                else:
                    pred_instances, _ = self.box_predictor.inference(predictions, proposals)
                    return pred_instances[0], r_indices[0]

            def get_conv5_features(self, features):
                assert len(self.in_features) == 1

                features = [features[f] for f in self.in_features]
                return features[0]

            def get_roi_features(self, features, proposals):
                features = [features[f] for f in self.in_features]  # removing the 'res5' key
                box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
                fc7, fc6 = self.box_head(box_features)

                return box_features, fc7, fc6


if __name__ == "__main__":

    #overwrite_detectron()


    config_path = 'mmexp/methods/bounding_boxes/config2.yaml'
    
    config = OmegaConf.load(os.path.dirname(os.getcwd()) + "/" + config_path)
    # Init encoder
    vision_module = build_image_encoder(config.image_encoder)

    # adjusting threshold
    # todo not working
    vision_module.cfg.defrost()
    vision_module.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.000001
    vision_module.cfg.freeze()

    # loading images
    general_path = "/imgs/protocol/"
    image_file_names = ["giraffe.jpg", "sheep.jpg"]
    # loading image
    images = []
    for img in image_file_names:
        img = cv2.imread(os.path.dirname(os.getcwd()) + general_path + img)

        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        #height, width = img.shape[:2]
        # loads the desired BGR
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1)) # correct shape
        print(img.shape)
        images.append(img)

    pooled = vision_module(images)
    print(pooled.shape)
    raise NotImplementedError
    inputs = [{"image": image} for image in images]
    vision_module.grid_feats_vqa.eval()
    images = vision_module.grid_feats_vqa.preprocess_image(inputs)
    features = vision_module.grid_feats_vqa.backbone(images.tensor)
    proposals, _ = vision_module.grid_feats_vqa.proposal_generator(images, features)
    # pooled features and box predictions
    box_features, pooled_features_fc7, pooled_features_fc6 = vision_module.grid_feats_vqa.roi_heads.get_roi_features(
        features, proposals)
    raise NotImplementedError
    print(pooled_features_fc7.shape)
    predictions = vision_module.grid_feats_vqa.roi_heads.box_predictor(pooled_features_fc7)
    #predictions, r_indices = vision_module.grid_feats_vqa.roi_heads.box_predictor(box_features)
    predictions, r_indices = vision_module.grid_feats_vqa.roi_heads.box_predictor.inference(predictions, proposals)
    #print(r_indices)
    #raise NotImplementedError
    box_type = type(proposals[0].proposal_boxes)
    proposal_bboxes = box_type.cat([p.proposal_boxes for p in proposals])
    proposal_bboxes.tensor = proposal_bboxes.tensor[r_indices]
    predictions[0].set("proposal_boxes", proposal_bboxes)
    # postprocess
    r = postprocessing.detector_postprocess(predictions[0], height, width)

    bboxes = r.get("proposal_boxes").tensor
    print(bboxes)


    #outputs = model.grid_feats_vqa([{"image": T.functional.to_tensor(img)}])
    # (img[:, :, ::-1] # convert RGB to BGR
    for img in images:
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=outputs[0]["instances"].pred_boxes)
        print(outputs)
        #print(outputs["instances"].pred_boxes)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(v_gt.get_image())
        plt.show()




