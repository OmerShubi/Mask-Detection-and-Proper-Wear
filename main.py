import torchvision
import torch
from utils import parse_data_for_model, parse_data_for_vis, show_images_and_bboxes
from config import train_dir, val_dir

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                             num_classes=2,
                                                             pretrained_backbone=False)
train_images, train_targets, train_imageid_toindex = parse_data_for_model(train_dir)
# test_images, test_targets, test_imageid_toindex = parse_data_for_model(test_dir)

output = model(train_images, train_targets)

model.eval()
train_predictions = model(train_images)

example_filenames = os.listdir(image_dir)
train_data = parse_data_for_vis(example_filenames)
show_images_and_bboxes(train_data, train_dir, train_imageid_toindex, train_predictions)
