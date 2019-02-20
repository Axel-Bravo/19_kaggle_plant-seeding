import numpy as np
import glob
import torch

from torch.utils.data import Dataset
from skimage import io, transform


class PlantDataset(Dataset):
    def __init__(self, data_path, transform=None):
        """
        Class to load the
        :param data_path:
        :param max_image_size:
        """
        # Initialization parameters
        self.data_path = data_path
        self.transform = transform

        # Calculation parameters
        self.images = glob.glob(data_path+'/**/*.png', recursive=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, image_i):
        image = io.imread(self.images[image_i])

        if self.transform:
            image = self.transform(image)

        return image
