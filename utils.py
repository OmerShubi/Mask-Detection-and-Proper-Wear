import os
import json
import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib import patches

def parse_data_for_model(image_dir):
    filenames = os.listdir(image_dir)
    images = []
    imageid_toindex = {}
    targets = []
    for index, filename in enumerate(filenames):
        image_id, bbox, proper_mask = filename.strip(".jpg").split("__")
        bbox = json.loads(bbox) #[x, y, w, h]
        bbox = torch.tensor([[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]]) #[x1, y1, x2, y2]
        proper_mask = torch.tensor([1]) if proper_mask.lower() == "true" else torch.tensor([0])
        im = torchvision.io.read_image(os.path.join(image_dir, filename)) #shape = (C,H,W) # TODO RGB/BGR?
        im = im / im.max()
        images.append(im)
        targets.append({"boxes": bbox, "labels": proper_mask})
        imageid_toindex[image_id] = index
    return images, targets, imageid_toindex

def parse_data_for_vis(image_dir):
    """
    Parse a directory with images.
    :param image_dir: Path to directory with images.
    :return: A list with (filename, image_id, bbox, proper_mask) for every image in the image_dir.
    """
    example_filenames = os.listdir(image_dir)
    data = []
    for filename in example_filenames:
        image_id, bbox, proper_mask = filename.strip(".jpg").split("__")
        bbox = json.loads(bbox)
        proper_mask = True if proper_mask.lower() == "true" else False
        data.append((filename, image_id, bbox, proper_mask))
    return data

def calc_iou(bbox_a, bbox_b):
    """
    Calculate intersection over union (IoU) between two bounding boxes with a (x, y, w, h) format.
    :param bbox_a: Bounding box A. 4-tuple/list.
    :param bbox_b: Bounding box B. 4-tuple/list.
    :return: Intersection over union (IoU) between bbox_a and bbox_b, between 0 and 1.
    """
    x1, y1, w1, h1 = bbox_a
    x2, y2, w2, h2 = bbox_b
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0.0 or h_intersection <= 0.0:  # No overlap
        return 0.0
    intersection = w_intersection * h_intersection
    union = w1 * h1 + w2 * h2 - intersection    # Union = Total Area - Intersection
    return intersection / union

def show_images_and_bboxes(data, image_dir, imageid_toindex, predictions):
    """
    Plot images with bounding boxes. Predicts random bounding boxes and computes IoU.
    :param data: Iterable with (filename, image_id, bbox, proper_mask) structure.
    :param image_dir: Path to directory with images.
    :return: None
    """
    # images, targets, images_id, filenames
    for filename, image_id, bbox, proper_mask in data:

        # Load image
        im = cv2.imread(os.path.join(image_dir, filename))
        # BGR to RGB
        im = im[:, :, ::-1]
        # Ground truth bbox
        x1, y1, w1, h1 = bbox
        # Predicted bbox
        predicted_bbox = predictions[imageid_toindex[image_id]]['boxes']
        predicted_left_bbox = predicted_bbox[predicted_bbox[:, 0].argmin()]
        x2, y2, w2, h2 = predicted_left_bbox[0], predicted_left_bbox[1], predicted_left_bbox[2]-predicted_left_bbox[0], predicted_left_bbox[3]-predicted_left_bbox[1]
        # Calculate IoU
        iou = calc_iou(bbox, (x2, y2, w2, h2))
        # Plot image and bboxes
        fig, ax = plt.subplots()
        ax.imshow(im)
        rect = patches.Rectangle((x1, y1), w1, h1,
                                 linewidth=2,
                                 edgecolor='g',
                                 facecolor='none',
                                 label='ground-truth')
        ax.add_patch(rect)
        rect = patches.Rectangle((x2, y2), w2, h2,
                                 linewidth=2,
                                 edgecolor='b',
                                 facecolor='none',
                                 label='predicted')
        ax.add_patch(rect)
        # fig.suptitle(f"proper_mask={proper_mask}, IoU={iou:.2f}")
        # ax.axis('off')
        fig.legend()
        plt.savefig(f'{image_id}_predicted.png')

