# 094295_hw2

#TODOS

runs:
0. if possible - solve iou, if not 
1. run with resnet mode
2. run with mobilenet mode
3. download csvs


report:
1. update num epochs x2
2. update achieved IOU and acc X2
3. run graphs again
4. add conclusions
5. export to html
6. submit to moodle

model:
1. Upload final model to drive
2. config update all

pre-submission check:   
1. clone from git
2. cd folder
3. create env
4. conda activate hw2_env
4. run `python predict.py test`
5. run `python evaluateResults.py` 

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
