from LightingModel import LitModel
import pytorch_lightning as pl
from maskData import MaskDataModule
import config as cfg
from pytorch_lightning.callbacks import ModelCheckpoint


pl.seed_everything(cfg.random_seed)
if __name__ == '__main__':
    model = LitModel()
    data_module = MaskDataModule(train_dir=cfg.train_dir, val_dir=cfg.val_dir,
                               batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_top_k=1, mode='max')

    trainer = pl.Trainer(max_epochs=cfg.max_epochs, gpus=1, num_sanity_val_steps=2, callbacks=[checkpoint_callback]
                     ,limit_train_batches=cfg.limit_train_batches, limit_val_batches=cfg.limit_val_batches)
    trainer.fit(model, datamodule=data_module)

