train_dir = "train"
val_dir = "test"
# TODO change all parameters
max_epochs = 2
batch_size = 4
DEBUG = True
if DEBUG:
    num_workers = 5 if not DEBUG else 0
model_path = 'lightning_logs/version_50/checkpoints/epoch=1-step=399.ckpt'
min_size_image = 400
max_size_image = 600