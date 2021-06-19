from torch.utils.data import Dataset, DataLoader
import torchvision
import torch


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
        self.filenames = os.listdir(image_dir)

        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.filenames.iloc[idx, 0]

        sample = self.parse_image(filename=img_name)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def parse_image(self, filename):
        image_id, bbox, proper_mask = filename.strip(".jpg").split("__")
        bbox = json.loads(bbox)  # [x, y, w, h]
        bbox = torch.tensor([[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]])  # [x1, y1, x2, y2]
        proper_mask = torch.tensor([1]) if proper_mask.lower() == "true" else torch.tensor([0])
        im = torchvision.io.read_image(os.path.join(image_dir, filename))  # shape = (C,H,W) # TODO RGB/BGR?
        im = im / im.max()
        target = {"boxes": bbox, "labels": proper_mask}

        return im, target
