import json
import os
from typing import Optional
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torch
import pytorch_lightning as pl
import config as cfg


class maskDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.filenames = sorted(os.listdir(image_dir))

        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.filenames[idx]

        sample = self.parse_image(filename=img_name)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def parse_image(self, filename):
        image_id, bbox, proper_mask = filename.strip(".jpg").split("__")
        bbox = json.loads(bbox)  # [x, y, w, h]
        bbox = torch.tensor([[bbox[cfg.x1_inx], bbox[cfg.y1_inx], bbox[cfg.x1_inx] + bbox[cfg.w_inx], bbox[cfg.y1_inx] + bbox[cfg.h_inx]]])  # [x1, y1, x2, y2]
        # true -> class 2, false -> class 1, background -> class 0
        proper_mask = torch.tensor([cfg.class_true]) if proper_mask.lower() == "true" else torch.tensor([cfg.class_false])
        im = torchvision.io.read_image(os.path.join(self.image_dir, filename))  # shape = (C,H,W)
        im = im / im.max()
        target = {"boxes": bbox, "labels": proper_mask}

        return im, target


class MaskDataModule(pl.LightningDataModule):

    def __init__(self, train_dir=None, val_dir=None, test_dir=None, batch_size=32, num_workers=0):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.transform = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train = maskDataset(self.train_dir, transform=self.transform)
            self.val = maskDataset(self.val_dir, transform=self.transform)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test = maskDataset(self.test_dir, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=custom_collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=custom_collate_fn, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=custom_collate_fn, num_workers=self.num_workers, shuffle=False)


def custom_collate_fn(batch):
    images = [x[0] for x in batch]
    targets = [x[1] for x in batch]
    return images, targets
