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

val_test_transforms = transforms.Compose([transforms.Resize((135,135)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Data sets
train_dataset = datasets.ImageFolder('data/train', train_transforms)
val_dataset = datasets.ImageFolder('data/val', val_test_transforms)
test_dataset = PlantDataset('data/test', val_test_transforms)

# Data Loader
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=12)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True, num_workers=12)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=12)


#%%
import torchvision
import numpy as np
import matplotlib.pyplot as plt

class_names = train_dataset.classes

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(val_loader))
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
for element in classes:
    print(class_names[element])

#%% 2| Train Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
