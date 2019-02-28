import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # CNN - Bock 1
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)

        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # MLP
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 12)

    def forward(self, x):
        x = F.relu(self.maxpool_1(self.conv_1(x)))
        x = F.relu(self.maxpool_1(self.conv_2(x)))
        x = F.relu(self.maxpool_1(self.conv_3(x)))

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def validation(model: nn.Module, val_loader: torch.utils.data.DataLoader, criterion: nn.CrossEntropyLoss, device: str)\
        -> (float, float):
    """
    Validation step during training
    :param model: NN model being trained
    :param val_loader: validation dataset loader
    :param criterion: loss criterion employed
    :param device: device where we execute the model; cuda/cpu
    :return: validation loss and validation accuracy
    """
    val_loss = 0
    val_accuracy = 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        val_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        val_accuracy += equality.type(torch.FloatTensor).mean()

    return val_loss, val_accuracy
