from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from torchvision.transforms import v2



class ImageDataset(Dataset):
    def __init__(self, img_dir, annotations_file):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir

        # assuming all images have same resolution
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[0, 0])
        image = decode_image(img_path)

        self.image_shape = image.shape
        self.channels, self.height, self.width = self.image_shape

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path).float() / 255.0
        label = self.img_labels.iloc[idx, 1]
        return image, label


class AugmentedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return self.transform(image), label
