import torch
import torch.nn as nn


class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvUnit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.drop = nn.Dropout2d()
        self.activ = nn.RReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.drop(output)
        output = self.activ(output)

        return output


class LinearUnit(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True):
        super(LinearUnit, self).__init__()

        self.fc = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.drop = nn.Dropout()

        self.activation = activation
        self.activ = nn.RReLU()

    def forward(self, input):
        output = self.fc(input)
        output = self.bn(output)
        output = self.drop(output)

        if self.activation:
            output = self.activ(output)

        return output


class SimpleNet(nn.Module):
    def __init__(self, num_classes=12):
        super(SimpleNet, self).__init__()

        # Create 14 layers of the unit with max pooling in between
        self.unit1 = ConvUnit(in_channels=3, out_channels=32)
        self.unit2 = ConvUnit(in_channels=32, out_channels=32)
        self.unit3 = ConvUnit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = ConvUnit(in_channels=32, out_channels=64)
        self.unit5 = ConvUnit(in_channels=64, out_channels=64)
        self.unit6 = ConvUnit(in_channels=64, out_channels=64)
        self.unit7 = ConvUnit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = ConvUnit(in_channels=64, out_channels=128)
        self.unit9 = ConvUnit(in_channels=128, out_channels=128)
        self.unit10 = ConvUnit(in_channels=128, out_channels=128)
        self.unit11 = ConvUnit(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = ConvUnit(in_channels=128, out_channels=128)
        self.unit13 = ConvUnit(in_channels=128, out_channels=128)
        self.unit14 = ConvUnit(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=4)

        # Add all the units into the Sequential layer in exact order
        self.cnn_net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5,
                                     self.unit6, self.unit7, self.pool2, self.unit8, self.unit9, self.unit10,
                                     self.unit11, self.pool3,self.unit12, self.unit13, self.unit14, self.avgpool)

        self.unit15 = LinearUnit(in_channels=128, out_channels=64)
        self.unit16 = LinearUnit(in_channels=64, out_channels=num_classes, activation=False)

        self.mlp_net = nn.Sequential(self.unit15, self.unit16)

    def forward(self, input):
        output = self.cnn_net(input)
        output = output.view(-1, 128)
        output = self.mlp_net(output)

        return output


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
