import torch
import torch.nn as nn


# Output size of conv = [(Input size + 2×Padding - Kernel size) / Stride] + 1
# Output size of max pool = [(Input size - Kernel size) / Stride] + 1
# The basic structure of this model
architecture_config = [
    # Kernel tuple (kernel_size, num_filters, stripe, padding)
    # Audio -> spectrum grayscale image -> conv1
    (3, 16, 1, 2),
    "MaxPool",
    (3, 32, 1, 2),
    (3, 128, 1, 2),
    "MaxPool",
    (3, 256, 1, 2),
    (1, 512, 1, 2),
    "MaxPool",
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNblock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv1(x)))


class SoundClassification(nn.Module):
    def __init__(self, in_channels=1, **kwargs):
        super(SoundClassification, self).__init__()
        self.architecture_config = architecture_config
        self.in_channels = in_channels
        self.net = self._create_conv_layers(self.architecture_config)

        # Calculate the output feature size:
        x = torch.rand((1, self.in_channels, 488, 488))
        with torch.no_grad():
            x = self.net(x)
            self.feature_size = x.shape[1] * x.shape[2] * x.shape[3]

        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        return self.net(x)

    def _create_conv_layers(self, architecture_config):
        layers = []
        in_channels = self.in_channels

        for x in architecture_config:
            if type(x) is tuple:
                layers += [
                    CNNblock(in_channels=in_channels,
                             out_channels=x[1],
                             kernel_size=x[0],
                             stride=x[2],
                             padding=x[3],
                             )
                ]
                in_channels = x[1]

            if type(x) is list:

                for j in range(x[:-1]):
                    for i in range(len(x) - 1):
                        conv = x[i]
                        layers += [
                            CNNblock(in_channels=in_channels,
                                     out_channels=conv[1],
                                     kernel_size=conv[0],
                                     stride=conv[0],
                                     padding=conv[3],)
                        ]
                        in_channels = conv[1]

            if type(x) is str:
                layers += [
                    nn.MaxPool2d(kernel_size=2, stride=2)
                ]

        return nn.Sequential(*layers)

    def _create_fcs(self, num_classes):
        out = num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.feature_size, out_features=1024),
            nn.Dropout(0.0),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=1024, out_features=out),
            nn.Softmax(dim=1),
        )
