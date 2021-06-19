import json
import os
from typing import Optional

from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torch
import pytorch_lightning as pl
from torchvision import transforms

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
        self.filenames = os.listdir(image_dir)

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
        bbox = torch.tensor([[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]])  # [x1, y1, x2, y2]
        proper_mask = torch.tensor([1]) if proper_mask.lower() == "true" else torch.tensor([0])
        im = torchvision.io.read_image(os.path.join(self.image_dir, filename))  # shape = (C,H,W) # TODO RGB/BGR?
        im = im / im.max()
        target = {"boxes": bbox, "labels": proper_mask}

        return im, target


class MaskDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.transform = None
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ]) # TODO add transformations?


    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            data = maskDataset(self.data_dir, transform=self.transform)
            self.train = data
            # self.train, self.val = random_split(data, [55000, 5000])
            # TODO add validation

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            # self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            pass # TODO
            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32, collate_fn=custom_collate_fn)

    # def val_dataloader(self):
    #     return DataLoader(self.val, batch_size=32)
    #
    # def test_dataloader(self):
    #     return DataLoader(self.test, batch_size=32)

def custom_collate_fn(batch):
    images = [x[0] for x in batch]
    targets = [x[1] for x in batch]
    return images, targets