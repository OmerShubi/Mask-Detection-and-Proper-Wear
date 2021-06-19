

import pandas as pd

from utils import show_images_and_bboxes2, parse_data_for_vis
from config import val_dir
pred = pd.read_csv('prediction.csv')
data = parse_data_for_vis(pred.loc[:,'filename'])
show_images_and_bboxes2(data, val_dir, pred)
