from LightingModel import LitModel
import pytorch_lightning as pl
from maskData import MaskDataModule
from config import train_dir

model = LitModel()
trainer = pl.Trainer()
trainer.fit(model, datamodule=MaskDataModule(data_dir=train_dir))

pass