train_dir = "train"
val_dir = "test"

# TODO change all parameters by debug/real running
DEBUG = True
if DEBUG:
    num_workers = 0
    max_epochs = 2
    batch_size = 4
    limit_train_batches = 0.05
    limit_val_batches = 0.05
    min_size_image = 100
    max_size_image = 100
else:
    num_workers = 5
    max_epochs = 100
    batch_size = 32
    limit_train_batches = 1.0
    limit_val_batches = 1.0
    min_size_image = 224
    max_size_image = 224
modes = {'mobilenet': 'mobilenet', 'resnet': 'resnet'}
mode = modes['resnet'] # True
model_path = 'lightning_logs/version_88/checkpoints/epoch=10-step=5499.ckpt'  # change to './model.ckpt' after upload
download_model = False
# Do not change
random_seed = 0
class_true = 2
class_false = 1
x1_inx = 0
y1_inx = 1
x2_inx = 2
y2_inx = 3
w_inx = 2
h_inx = 3
