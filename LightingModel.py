import torch

import pytorch_lightning as pl
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, mobilenet_backbone
from torchvision.ops import MultiScaleRoIAlign

from utils import calc_iou

import config as cfg


class LitModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.mode = cfg.mode
        self.min_size_image = cfg.min_size_image
        self.max_size_image = cfg.max_size_image
        pretrained = False
        num_classes = 3
        if self.mode == 'resnet':
            backbone = resnet_fpn_backbone('resnet50',
                                           pretrained)

            self.model = FasterRCNN(backbone,
                                    num_classes,
                                    min_size=cfg.min_size_image,
                                    max_size=cfg.max_size_image,
                                    box_detections_per_img=1)

        elif self.mode == 'mobilenet':
            backbone = mobilenet_backbone(backbone_name="mobilenet_v3_large",
                                          pretrained=pretrained,
                                          fpn=True)

            anchor_generator = AnchorGenerator(
                sizes=(32, 64, 128),
                aspect_ratios=(0.5, 1.0, 2.0))

            box_roi_pooler = MultiScaleRoIAlign(
                featmap_names=['0'], output_size=7, sampling_ratio=2)

            self.model = FasterRCNN(backbone,
                                    num_classes,
                                    rpn_anchor_generator=anchor_generator,
                                    box_roi_pool=box_roi_pooler,
                                    min_size=cfg.min_size_image,
                                    max_size=cfg.max_size_image,
                                    box_detections_per_img=1)

        else:
            print("Model type not supported")

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        loss, iou, acc = self.step(batch)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_iou', iou, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss, 'iou': iou, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        # validation_step defines the train loop. It is independent of forward
        loss, iou, acc = self.step(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_iou', iou, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_sum', acc+iou, on_epoch=True, prog_bar=False, logger=True)
        return {'loss': loss, 'iou': iou, 'acc': acc, 'val_sum': acc+iou}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def calc_metrics(self, loss, detections, targets):
        sum_loss = sum(loss.values())
        iou = 0
        acc = 0
        for detection, target in zip(detections, targets):
            pred_bbox = detection['boxes']
            pred_label = detection['labels']
            if pred_bbox.numel():
                pred_bbox = pred_bbox[0]
                pred_bbox[cfg.w_inx] = pred_bbox[cfg.x2_inx] - pred_bbox[cfg.x1_inx]
                pred_bbox[cfg.h_inx] = pred_bbox[cfg.y2_inx] - pred_bbox[cfg.y1_inx]
                true_bbox = target['boxes'][0].tolist()
                true_bbox[cfg.w_inx] = true_bbox[cfg.x2_inx] - true_bbox[cfg.x1_inx]
                true_bbox[cfg.h_inx] = true_bbox[cfg.y2_inx] - true_bbox[cfg.y1_inx]
                iou += calc_iou(pred_bbox, true_bbox)

                if pred_label == target['labels']:
                    acc += 1

        if isinstance(iou, torch.Tensor):
            iou = iou.item()
        return sum_loss, iou, acc

    def step(self, batch):
        images, targets = batch
        to_remove_indices = []
        for indx, target in enumerate(targets):
            bbox = target['boxes'][0]
            if bbox[cfg.x1_inx] >= bbox[cfg.x2_inx] or bbox[cfg.y1_inx] >= bbox[cfg.y2_inx]:
                to_remove_indices.append(indx)
        for indx in to_remove_indices:
            images.pop(indx)
            targets.pop(indx)

        detections, losses = self.model(images, targets)
        loss, iou, acc = self.calc_metrics(losses, detections, targets)
        acc = acc / len(images)
        iou = iou / len(images)
        return loss, iou, acc
