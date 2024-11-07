
import torch
import torch.nn as nn



'''Basic structure of yolo'''
architecture_config = [
    # Kernel tuple (kernel_size, num_filters, stride, padding)
    (7, 64, 3, 2),
    # A 2D MaxPool will be added
    "MaxPool",
    (3, 192, 1, 1),
    "MaxPool",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "MaxPool",
    # List: [kernel_1, kernel_2, i]
    # The same kernel combination repeats i times
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "MaxPool",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


def test(split_size, num_boxes, num_classes):
    model = Yolov1(split_size=split_size,
                   num_boxes=num_boxes,
                   num_classes=num_classes)
    x = torch.randn((2, 3, 488, 488))
    print(model(x).shape)


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm1(self.conv1(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture_config = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layer(self.architecture_config)

        # Calculate the feature dimension
        x = torch.randn((1, in_channels, 488, 488))
        with torch.no_grad():
            x = self.darknet(x)
        self.feature_size = x.shape[1] * x.shape[2] * x.shape[3]

        self.fcs = self._create_fcs(**kwargs)


    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layer(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) is tuple:
                layers += [
                    CNNBlock(in_channels=in_channels,
                                   out_channels=x[1],
                                   kernel_size=x[0],
                                   stride=x[2],
                                   padding=x[3],)
                    ]
                in_channels = x[1]
            elif type(x) is str:
                layers += [
                    nn.MaxPool2d(kernel_size=2, stride=2)
                ]
            elif type(x) is list:
                conv1 = x[0] # The kernel tuple
                conv2 = x[1] # The kernel tuple
                num_repeats = x[2] # int

                for i in range(num_repeats):
                    layers += [
                        CNNBlock(in_channels=in_channels,
                                 out_channels=conv1[1],
                                 kernel_size=conv1[0],
                                 stride=conv1[2],
                                 padding=conv1[3],)
                    ]

                    layers += [
                        CNNBlock(in_channels=conv1[1],
                                 out_channels=conv2[1],
                                 kernel_size=conv2[0],
                                 stride=conv2[2],
                                 padding=conv2[3], )
                    ]
                    # Update channel for next layer
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        s, b, c = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(4096, s * s * (c + b * 5)),
        )


if __name__ == '__main__':
    test(split_size=7, num_boxes=2, num_classes=20)