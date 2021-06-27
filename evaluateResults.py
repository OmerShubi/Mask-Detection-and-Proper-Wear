import os
import pandas as pd
from utils import parse_data_for_vis, calc_iou
import config as cfg

print(f"model {cfg.model_path} is ")
pred = pd.read_csv('prediction.csv')
data = pd.DataFrame(parse_data_for_vis(sorted(os.listdir('test'))),
                    columns=['filename', 'id', 'bbox', 'proper_mask'])
iou = 0
for pred_bbox, true_bbox in zip(pred[['x','y','w','h']].values, data["bbox"].values):
    iou += calc_iou(pred_bbox, true_bbox)
print(f"iou : {iou/len(data)}")
acc = 0
for pred_proper_mask, true_proper_mask in zip(pred["proper_mask"].values, data["proper_mask"].values):
    acc += (pred_proper_mask == true_proper_mask)
print(f"acc : {acc/len(data)}")
