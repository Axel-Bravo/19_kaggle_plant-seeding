#%% 0| Import and function declaration
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler

from data_process import PlantDataset
from model import CNNModel
from train_module import train_model

# Parameters
batch_size = 64
num_epochs = 100
learning_rate = 0.025


#%% 1| Load dataset
# Transformers
train_transforms = transforms.Compose([transforms.RandomResizedCrop(50),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

val_test_transforms = transforms.Compose([transforms.Resize(56),
                                          transforms.CenterCrop(50),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Data sets
train_dataset = datasets.ImageFolder('data/train_split', train_transforms)
val_dataset = datasets.ImageFolder('data/val_split', val_test_transforms)
test_dataset = PlantDataset('data/test', val_test_transforms)

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
train_dataset_augmenter = 2

# Data Loader
train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=12)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

train_data_loader = {'train': train_loader, 'val': val_loader}


#%% 2| Model - Preparation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CNNModel()
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  # Observe that all parameters are being optimized
lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)  # Decay LR by a factor of 0.1 every 7 epochs


#%% 3| Model - Training

model_trained = train_model(device=device, data_loaders=train_data_loader, dataset_sizes=dataset_sizes, model=model,
                            criterion=criterion, optimizer=optimizer_ft, scheduler=lr_scheduler, num_epochs=num_epochs)


#%% 4| model -Save
torch.save(model_trained.state_dict(), 'model/trained.pth')
