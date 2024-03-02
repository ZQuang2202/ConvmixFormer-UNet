import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class ISIC(Dataset):
  def __init__(self, image_dir, mask_dir, transform=None, state=None):
    self.image_dir = image_dir
    self.mask_dir = mask_dir
    self.transform = transform
    self.images = os.listdir(image_dir)

  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
    img_path = os.path.join(self.image_dir, self.images[index])
    mask_path = os.path.join(self.mask_dir, self.images[index]).replace(".bmp","_anno.bmp")
    image = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path))
    mask[mask >= 1] = 1.0

    if self.transform is not None:
      augmentations = self.transform(image=image, mask=mask)
      image = augmentations["image"]
      mask = augmentations["mask"]
    return image, mask


class DataScienceBowl(Dataset):
  def __init__(self, x_train_dir, y_train_dir, transform=None):

    self.images = np.load(x_train_dir)
    self.masks = np.load(y_train_dir)
    self.transform = transform

  def __len__(self):
    return self.images.shape[0]

  def __getitem__(self, index):
    image = self.images[index]
    mask = self.masks[index].squeeze()

    if self.transform is not None:
      augmentations = self.transform(image=image, mask=mask)
      image = augmentations["image"]
      mask = augmentations["mask"]
    return image, mask


class DataSegmenModule():
    def __init__(self, x_train_dir, y_train_dir, x_val_dir, y_val_dir, x_test_dir, y_test_dir):
        self.x_train_dir = x_train_dir
        self.y_train_dir = y_train_dir
        self.x_val_dir = x_val_dir
        self.y_val_dir = y_val_dir
        self.x_test_dir = x_test_dir
        self.y_test_dir = y_test_dir

        IMAGE_HEIGHT = 256
        IMAGE_WIDTH = 256

        self.train_transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Rotate(limit=20, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(
                    #mean = (0.485, 0.456, 0.406),
                    #std = (0.229, 0.224, 0.225),
                    mean = (0., 0., 0.),
                    std = (1., 1., 1.),
                    max_pixel_value = 255.0
                ),
                ToTensorV2(),
            ])

        self.val_transform = A.Compose(
            [
                #A.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(
                    mean = (0., 0., 0.),
                    std = (1., 1., 1.),
                    max_pixel_value = 255.0
                ),
                ToTensorV2(),
            ])
    def train_loader(self, batch_size, num_workers=8, shuffle=True):
        train_data = DataScienceBowl(self.x_train_dir,self.y_train_dir,
                                     transform=self.train_transform)
        train_loader = DataLoader(
            train_data,
            batch_size = batch_size,
            num_workers = num_workers,
            shuffle=True,
        )
        return train_loader

    def val_loader(self, batch_size, num_workers=8, shuffle=False):
        val_data = DataScienceBowl(self.x_val_dir,
                                   self.y_val_dir,
                                   transform=self.val_transform)
        val_loader = DataLoader(
            val_data,
            batch_size=1,
            num_workers = num_workers,
            shuffle=False,
        )
        return val_loader

    def test_loader(self, batch_size, num_workers=8, shuffle=False):
        test_data = DataScienceBowl(self.x_test_dir, self.y_test_dir)
        test_loader = DataLoader(
            test_data,
            batch_size = 1,
            num_workers = num_workers,
            shuffle=False,
        )
        return test_loader
    