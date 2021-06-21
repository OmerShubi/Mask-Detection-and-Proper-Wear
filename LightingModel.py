import torch

import pytorch_lightning as pl
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from utils import calc_iou


class LitModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        # self.save_hyperparameters() # TODO add if have hyper parms?

        trainable_backbone_layers = 5
        pretrained_backbone = False
        num_classes = 2 # TODO should be 1 or 2?
        backbone = resnet_fpn_backbone('resnet18', pretrained_backbone, trainable_layers=trainable_backbone_layers)
        self.model = FasterRCNN(backbone, num_classes, box_detections_per_img=1, min_size=400, max_size=600) # TODO define as parameter
        # self.model = fasterrcnn_resnet50_fpn(pretrained=False,
        #                                      num_classes=2,
        #                                      pretrained_backbone=False, box_detections_per_img=1)


        # todo add test_step?
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions

        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        # TODO deal with empty bboxes okay? in test?
        images, targets = batch
        to_remove_indices = []
        for indx,target in enumerate(targets):
            bbox = target['boxes'][0]
            if bbox[0]>=bbox[2] or bbox[1] >= bbox[3]:
                to_remove_indices.append(indx)
        for indx in to_remove_indices:
            images.pop(indx)
            targets.pop(indx)

        detections, losses = self.model(images, targets) # TODO detections is empty (change source)
        loss= sum(losses.values())
        # loss, iou, acc = self.calc_metrics(losses, detections, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_iou', iou, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}


    # def validation_step(self, batch, batch_idx):
    #     # training_step defines the train loop. It is independent of forward
    #     images, targets = batch
    #     detections, losses = self.model(images, targets) # TODO loss is empty (change source)
    #     loss, iou, acc = self.calc_metrics(losses, detections, targets)
    #     self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     self.log('val_iou', iou, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     return {'loss': loss}


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
            if pred_bbox:
                pred_bbox[2] -= pred_bbox[0]
                pred_bbox[3] -= pred_bbox[1]
                iou += calc_iou(pred_bbox, target['boxes'][0].tolist())

            acc += (pred_label==target['labels'])

        return sum_loss, iou, acc
