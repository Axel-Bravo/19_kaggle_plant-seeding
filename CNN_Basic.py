#%% Import and function declaration
# 0| Load
from torch.utils.data import Dataset, DataLoader
import glob
from skimage import io, transform

#%% Load dataset
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

#%% Testing

train_data = PlantDataset(data_path='data/train')

# TODO: implement transforms https://pytorch.org/tutorials/beginner/data_loading_tutorial.html