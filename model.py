import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # CNN - Bock 1
        self.conv_1_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv_1_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3)

        # CNN - Block 2
        self.conv_2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv_2_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3)

        # MLP
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 12)

    def forward(self, x):
        x = F.relu(self.maxpool_1(self.conv_1_2(self.conv_1_1(x))))
        x = F.relu(self.maxpool_1(self.conv_2_2(self.conv_2_1(x))))

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