import torch

import pytorch_lightning as pl
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, mobilenet_backbone

from utils import calc_iou

import config as cfg

class LitModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        trainable_backbone_layers = 5
        pretrained = False
        num_classes = 3
        backbone = resnet_fpn_backbone('resnet18',
                                       pretrained,
                                       trainable_layers=trainable_backbone_layers)
        # backbone = mobilenet_backbone(backbone_name="mobilenet_v3_small",
        #                               pretrained=pretrained,
        #                               fpn = False,
        #                               trainable_layers=trainable_backbone_layers)
        self.model = FasterRCNN(backbone,
                                num_classes,
                                min_size=cfg.min_size_image,
                                max_size=cfg.max_size_image,
                                box_detections_per_img=1)

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

        return {'loss': loss, 'iou':iou, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        # validation_step defines the train loop. It is independent of forward
        loss, iou, acc = self.step(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_iou', iou, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'iou':iou, 'acc': acc}

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
                iou += calc_iou(pred_bbox, target['boxes'][0].tolist())

                if pred_label == target['labels']:
                    acc += 1

        if isinstance(iou, torch.Tensor):
            iou = iou.item()
        return sum_loss, iou, acc

    # def calc_metrics(self, loss, detections, targets):
    #     sum_loss = sum(loss.values())
    #     iou = 0
    #     acc = 0
    #     for detection, target in zip(detections, targets):
    #         pred_bbox = detection['boxes']
    #         if pred_bbox.numel():
    #             left_bbox_inx = pred_bbox[:, cfg.x1_inx].argmin()
    #             pred_bbox = pred_bbox[left_bbox_inx]
    #             pred_label = detection['labels'][left_bbox_inx]
    #             # pred_bbox = pred_bbox[0]
    #             pred_bbox[cfg.w_inx] = pred_bbox[cfg.x2_inx] - pred_bbox[cfg.x1_inx]
    #             pred_bbox[cfg.h_inx] = pred_bbox[cfg.y2_inx] - pred_bbox[cfg.y1_inx]
    #             iou += calc_iou(pred_bbox, target['boxes'][0].tolist())
    #
    #             if pred_label==target['labels']:
    #                 acc += 1
    #
    #     if isinstance(iou, torch.Tensor):
    #         iou = iou.item()
    #     return sum_loss, iou, acc


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

        # x = self.model(images, targets)
        detections, losses = self.model(images, targets)
        # loss= sum(losses.values())
        loss, iou, acc = self.calc_metrics(losses, detections, targets)
        acc = acc / len(images)
        iou = iou / len(images)
        return loss, iou, acc