import os
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import random
from LightingModel import LitModel
from config import model_path
# Parsing script arguments
from maskData import MaskDataModule, maskDataset

parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Reading input folder
files = os.listdir(args.input_folder)

#####
model = LitModel.load_from_checkpoint(model_path)
model.eval()
dm = MaskDataModule(test_dir=args.input_folder)
dm.setup(stage='test')
data_loader = dm.test_dataloader()
preds = []
for (images, targets) in data_loader:
    x=model(images)
    preds.extend(x[0])

proper_mask_pred = []
for pred in preds:
    if len(pred['labels']) > 0:
        mask_pred = int(pred['labels'])==1
        proper_mask_pred.append(mask_pred)
    else:
        proper_mask_pred.append(False)
bbox_pred = []
for inx, pred in enumerate(preds):
    try:
        bbox = pred['boxes'][0].tolist()
    except IndexError:
        height = data_loader.dataset.__getitem__(0)[0].shape[2]
        width = data_loader.dataset.__getitem__(0)[0].shape[1]
        bbox = [random.randint(0, height), random.randint(0, width)]
        bbox.extend([random.randint(bbox[0], height), random.randint(bbox[1], width)])

    bbox[2] = bbox[2] - bbox[0]
    bbox[3] = bbox[3] - bbox[1]
    bbox_pred.append(bbox)
pd.DataFrame([files])

prediction_df = pd.DataFrame(zip(files, *np.array(bbox_pred, dtype=int).transpose(), proper_mask_pred),
                             columns=['filename', 'x', 'y', 'w', 'h', 'proper_mask'])
####

prediction_df.to_csv("prediction.csv", index=False, header=True)
