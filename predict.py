import os
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import random
from LightingModel import LitModel
import config as cfg
# Parsing script arguments
from maskData import MaskDataModule, maskDataset
import torch

from utils import download_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process input')
    parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
    args = parser.parse_args()

    if cfg.download_model:
        download_model(file_id='123', dest_path='./model.zip') # TODO change file_id!

    model = LitModel.load_from_checkpoint(cfg.model_path)
    model.eval()
    if torch.cuda.is_available():
        model.to(device=torch.device("cuda"))
    dm = MaskDataModule(test_dir=args.input_folder, batch_size=4)
    dm.setup(stage='test')
    data_loader = dm.test_dataloader()

    proper_mask_pred = []
    count_no_prediction = 0
    bbox_pred = []
    for inx, (images, targets) in enumerate(data_loader):
        if torch.cuda.is_available():
            images = [img.to(device=torch.device("cuda")) for img in images]
        predictions, _ = model(images)
        print(f"Finished predict batch {inx}")
        for pred in predictions:
            if len(pred['labels']) > 0:
                mask_pred = int(pred['labels']) == cfg.class_true
                proper_mask_pred.append(mask_pred)
            else:  # Return false label if empty label
                proper_mask_pred.append(False)
                count_no_prediction += 1
            try:
                bbox = pred['boxes'][0].tolist()
            except IndexError:
                height = data_loader.dataset.__getitem__(0)[0].shape[2]
                width = data_loader.dataset.__getitem__(0)[0].shape[1]
                # Return random prediction when empty prediction
                bbox = [random.randint(0, height), random.randint(0, width)]
                bbox.extend([random.randint(bbox[cfg.x1_inx], height), random.randint(bbox[cfg.y1_inx], width)])

            bbox[cfg.w_inx] = bbox[cfg.x2_inx] - bbox[cfg.x1_inx]
            bbox[cfg.h_inx] = bbox[cfg.y2_inx] - bbox[cfg.y1_inx]
            bbox_pred.append(bbox)
        print(f"Finished saving predictions batch {inx}")

    print(f"Random predict {count_no_prediction} out of {len(proper_mask_pred)}")
    files = data_loader.dataset.filenames
    prediction_df = pd.DataFrame(zip(files, *np.array(bbox_pred, dtype=int).transpose(), proper_mask_pred),
                                 columns=['filename', 'x', 'y', 'w', 'h', 'proper_mask'])

    prediction_df.to_csv("prediction.csv", index=False, header=True)

