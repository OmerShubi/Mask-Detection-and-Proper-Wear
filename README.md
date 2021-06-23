# 094295_hw2


#TODOS
- code todos
- second experiment : backbone, hypeprarmeters , transformations, change - im = im / im.max()
- finish report : results : acc, iou in chosen epoch, vis some predictions, second experiment, graphs in both. 
- tries if can fall
before submission - 
- save & load model from google drive
- check env creation and running
- README
- read project def
- public git
- Hide code cells and export report to html
2. make repo public
3. delete unwanted files from repo and save pretrained model, scalers and such   
4. recreate env and test model

python code for mask detection and for classifying proper wear.

Detailed report can be found under `report` file.

To get the code simply clone it - 
`git clone https://github.com/scaperex/094295_hw2.git`

Then, to setup the environment - 
- `cd 094295_hw2`
- `conda env create -f environment.yml`

Activate it by -
`conda activate hw2_env`

Finally, to evaluate the pretrained model simply run 
`python predict.py <PATH_TO_FOLDER>`.

This saves a `prediction.csv` file with the predictions. 

Additionally, the code consists of -
 - Training the model uses the `train.py` script. This makes use of the model as defined in  `LightningModel.py` and by the parameters  given by `config.py`. The data loading process is defined in `maskData.py`
 -  For visualing the prediction results run `visualizeResults.py` with the matching parameters.

Note, the data is expected to be given in the same format as used for training.


To view tensorboard logs:

`tensorboard --logdir lightning_logs --bind_all`

Then in browser:
`<PUBLIC_IP>:<PORT>`
