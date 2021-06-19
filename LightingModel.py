import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class LitModel(pl.LightningModule):

    def __init__(self):
        super().__init__()

        trainable_backbone_layers = 5
        pretrained_backbone = False
        num_classes = 2 # TODO should be 1 or 2?
        backbone = resnet_fpn_backbone('resnet18', pretrained_backbone, trainable_layers=trainable_backbone_layers)
        self.model = FasterRCNN(backbone, num_classes, box_detections_per_img=1)
        # self.save_hyperparameters()
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
        images, targets = batch
        x_hat = self.model(images, targets)
        loss = sum(x_hat.values())
        self.log('train_loss', loss)
        return loss


    # def validation_step(self, batch, batch_idx):
    #     # training_step defines the train loop. It is independent of forward
    #     images, targets = batch
    #     x_hat = self.model(images, targets)
    #     loss = x_hat['loss_objectness'] # TODO
    #     self.log('val_loss', loss)
    #     return loss



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer