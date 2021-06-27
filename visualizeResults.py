import os
import pandas as pd
from utils import show_images_and_bboxes, parse_data_for_vis, calc_iou
import config as cfg
from subprocess import call

# prediction.csv suppose to be on image_dir
# call(["python", "predict.py", "test"])
print(f"model {cfg.model_path} is ")
pred = pd.read_csv('prediction.csv')
data = pd.DataFrame(parse_data_for_vis(os.listdir('test_example')),
                    columns=['filename', 'id', 'bbox', 'proper_mask'])
iou = 0
for pred_bbox, true_bbox in zip(pred[['x','y','w','h']].values, data["bbox"].values):
    iou += calc_iou(pred_bbox, true_bbox)
print(f"iou : {iou/len(data)}")
acc = 0
for pred_proper_mask, true_proper_mask in zip(pred["proper_mask"].values, data["proper_mask"].values):
    acc += pred_proper_mask == true_proper_mask
print(f"acc : {acc/len(data)}")

image_dir = 'test'
data_for_vis = parse_data_for_vis(pred.loc[:,'filename'])
show_images_and_bboxes(parse_data_for_vis(pred.loc[:,'filename']), image_dir, pred)
