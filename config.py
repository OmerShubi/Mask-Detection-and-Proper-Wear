train_dir = "train"
val_dir = "test"
max_epochs = 2
batch_size = 4
DEBUG =True
if DEBUG:
    num_workers = 5 if not DEBUG else 0
model_path = 'lightning_logs/version_18/checkpoints/epoch=1-step=7999.ckpt' # TODO replace with final model
model_path = 'lightning_logs/version_19/checkpoints/epoch=0-step=3999.ckpt' # TODO replace with final model
model_path = 'lightning_logs/version_20/checkpoints/epoch=1-step=7999.ckpt' # TODO replace with final model
