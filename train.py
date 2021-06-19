from LightingModel import LitModel
import pytorch_lightning as pl
from maskData import MaskDataModule
import config as cfg

model = LitModel()
trainer = pl.Trainer(max_epochs=cfg.max_epochs)
trainer.fit(model, datamodule=MaskDataModule(data_dir=cfg.train_dir))

pass