#%% 0| Import and function declaration
from data_process import PlantDataset
from model import CNNModel, validation

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Parameters
batch_size = 64
num_epochs = 30
learning_rate = 0.001


#%% 1| Load dataset
# Transformers
train_transforms = transforms.Compose([transforms.RandomResizedCrop(135),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

val_test_transforms = transforms.Compose([transforms.Resize((135, 135)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Data sets
train_dataset = datasets.ImageFolder('data/train_split', train_transforms)
val_dataset = datasets.ImageFolder('data/val_split', val_test_transforms)
test_dataset = PlantDataset('data/test', val_test_transforms)

# Data Loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12)


#%% 2| Model - Preparation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CNNModel()
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#%%% 3 Model - Training

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Clear gradients w.r.t. parameters
        outputs = model(images)  # Forward pass to get output
        loss = criterion(outputs, labels)  # Calculate Loss: softmax --> cross entropy loss
        loss.backward()  # Getting gradients w.r.t. parameters
        optimizer.step()  # Updating parameters

    # Evaluation of trainning
    model.eval()  # Make sure network is in eval mode for inference

    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        test_loss, accuracy = validation(model, val_loader, criterion, device)

    print("Epoch: {}/{}.. ".format(epoch + 1, num_epochs),
          "Training Loss: {:.3f}.. ".format(loss),
          "Val Loss: {:.3f}.. ".format(test_loss / len(test_loader)),
          "Val Accuracy: {:.3f}".format(accuracy / len(test_loader)))

    model.train()  # Make sure training is back on

