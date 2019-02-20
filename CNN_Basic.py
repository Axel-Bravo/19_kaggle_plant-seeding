#%% Import and function declaration
# 0| Load
from data_process import PlantDataset, Rescale, RandomCrop, ToTensor

from torchvision import transforms
from torch.utils.data import DataLoader

#%% Load dataset
train_data = PlantDataset(data_path='data/train',
                          transform=transforms.Compose([Rescale(135), RandomCrop(224), ToTensor()]))

dataloader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)
