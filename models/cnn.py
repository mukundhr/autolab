import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, filters=32, num_layers=2):
        super().__init__()

        layers = []
        in_channels = 1

        # build convolution blocks dynamically
        for _ in range(num_layers):

            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=3,
                    padding=1
                )
            )

            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))

            in_channels = filters

        self.conv = nn.Sequential(*layers)

        # compute feature size automatically
        self.feature_size = self._get_feature_size()

        self.fc = nn.Linear(self.feature_size, 10)

    def _get_feature_size(self):

        with torch.no_grad():

            x = torch.zeros(1, 1, 28, 28)

            x = self.conv(x)

            return x.view(1, -1).size(1)

    def forward(self, x):

        x = self.conv(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x