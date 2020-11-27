import json
import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T


class ImageLoader():
    def __init__(self, data_dir, labels=None, batch_size=64,
                 rotation=30, p_flip=0.3, resize=256, crop=224,
                 means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
        
        self.data_dir = data_dir
        self.folders = ['train', 'valid', 'test']
        self.batch_size = batch_size
        self.means = means
        self.stds = stds
        self.rotation = rotation
        self.p_flip = p_flip
        self.resize = resize
        self.crop = crop
        
        self.transforms = None
        self.datasets = None
        self.dataloader = None
        self.class_to_labels = labels
        self.class_to_idx = None
        
        self._set_transforms()
        self._load_datasets()
        self._initialize_dataloader()
    
    def _set_transforms(self, transforms=None):
        if transforms is None:
            transforms = {
                'train': T.Compose([T.RandomRotation(self.rotation), 
                                    T.RandomResizedCrop(self.crop),
                                    T.ToTensor(),
                                    T.Normalize(mean=self.means,
                                                std=self.stds)]),
                'valid': T.Compose([T.Resize(self.resize),
                                    T.CenterCrop(self.crop),
                                    T.ToTensor(),
                                    T.Normalize(mean=self.means,
                                                std=self.stds)]),
                'test': T.Compose([T.Resize(self.resize),
                                   T.CenterCrop(self.crop),
                                   T.ToTensor(),
                                   T.Normalize(mean=self.means,
                                               std=self.stds)])
            }
        self.transforms = transforms
            
    def _load_datasets(self):
        t = self.transforms
        datasets = {
            folder: ImageFolder(Path(self.data_dir) / folder, transform=t[folder]) for folder in self.folders
        }
        self.datasets = datasets
        self.class_to_idx = datasets[self.folders[0]].class_to_idx
    
    def _initialize_dataloader(self):
        d = self.datasets
        dataloader = {
            folder: DataLoader(d[folder], batch_size=self.batch_size, shuffle=True) for folder in self.folders
        }
        self.dataloader = dataloader
    
    def load(self):
        return self.dataloader

def draw_random_image(data, show=False):
    imgs = [path for (path, _) in data.imgs]
    img_path = np.random.choice(imgs)
    return img_path

def labels_from_json(filename='cat_to_name.json'):
    with open(filename, 'r') as f:
        return json.load(f)

def image_as_tensor(image, resize=256, crop=224,
                    means=[0.485, 0.456, 0.406],
                    stds=[0.229, 0.224, 0.225]):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a PyTorch Tensor.
    '''
    # Load and resize
    image = Image.open(image)
    image.thumbnail((resize, resize))
    
    # Define Coordinates for cropping
    width, height = image.size
    
    left = (width - crop) // 2
    upper = (height - crop) // 2
    right = (width + crop) // 2
    lower = (height + crop) // 2
    
    box = (left, upper, right, lower)
    
    # Crop
    image = image.crop(box)
    
    # Convert to numpy
    image = np.array(image) / 255
    image = (image - means) / stds
    image = image.transpose(2, 0, 1)

    return torch.from_numpy(image).unsqueeze(0)