import pandas as pd
from utils import show_images_and_bboxes, parse_data_for_vis
import config as cfg

# prediction.csv suppose to be on image_dir
pred = pd.read_csv('prediction.csv')
data = parse_data_for_vis(pred.loc[:,'filename'])
image_dir = 'test_example'
show_images_and_bboxes(data, image_dir, pred)
