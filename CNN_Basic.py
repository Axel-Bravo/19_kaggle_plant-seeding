#%% 0| Import and function declaration
from data_process import PlantDataset

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

#%% 1| Load dataset
# Transformers
train_transforms = transforms.Compose([transforms.RandomResizedCrop(135),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(135),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Data sets
train_dataset = datasets.ImageFolder('data/train', train_transforms)
test_dataset = PlantDataset('data/test', test_transforms)

# Data Loader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=12)


#%% 2| Train Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
