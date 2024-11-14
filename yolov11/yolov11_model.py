
import torch
import torch.nn as nn
import torch.nn.functional as F



'''Global Definition Of Repeat factor'''
R = 3
'''Input image: RGB, 640 * 640'''
# Output size of conv = [(Input size + 2×Padding - Kernel size) / Stride] + 1
# Output size of max pool = [(Input size - Kernel size) / Stride] + 1
'''About the list in the config dic: [operation_name, (parameter0, parameter1, ...), repeat_num]'''
back_bone_architecture = {
    # CNN layer tuple (input_channel, out_channels, kernel, stride, padding)
    # CNN layer list with repeat num ['operation_name', (in, out, k, s, p), repeat_num]
    ['Conv', (3, 64, 3, 2, 1), 1],
    ['Conv', (64, 128, 3, 2, 1), 1],
    ['C3K2', (128, 128, 3, 1, 1), 3 * R],
    ['Conv', (128, 256, 3, 2, 1), 1],
    ['C3K2',(256, 256, 3, 1, 1), 6 * R],
    ['Conv', (256, 512, 3, 2, 1), 1],
    ['C3K2', (512, 512, 3, 1, 1), 6 * R],
    ['Conv', (512, 1024, 3, 2, 1), 1]
}

neck_architecture = {
    ['SPFF', 1],
    ['C2PSA', 1],
    # Up sample list: ['operation_name', output_size(D, H, W), scale_factor]
    ['Upsample', (512, 40, 40), 1],
    # Concat list: ['operation_name', dimension, (tensor1, tensor2)]
    ["Concat", 2,],
    ['C3K2', (512, 512, 3, 1, 1), 3 * R],
    ['Upsample', (256, 80, 80), 1],
    ["Concat", (2), 1],
    ['C3K2', (256, 256, 3, 2, 1), 3 * R],
    ['Conv', (256, 512, 3, 2, 1), 1],
    ["Concat", (2), 1],
    ['C3K2', (512, 512, 3, 2, 1), 3 * R],
    ['Conv', (512, 1024, 3, 2, 1), 1],
    ['Concat', (2), 1],
    ['C3K2', (1024, 1024, 3, 1, 1), 3 * R]
}


class CNNblock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNblock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, *kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class SPFF(nn.Module):
    def __init__(self, in_channels_list, out_channels, *kwargs):
        super(SPFF, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.conv_list = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)
             for in_channels in self.in_channels_list]
        )
        self.refined_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, **kwargs)

    def forward(self, feature_maps):
        if not feature_maps:
            raise ValueError('NO input feature maps provided to the SPFF layer')

        max_size = feature_maps[0].size()[2:]
        processed_features = []

        for i, feat in enumerate(feature_maps):
            resized_feat = F.interpolate(feat, size=max_size, mode='bilinear', align_corners=False)
            processed_feat = self.conv_list[i](resized_feat)
            processed_features.append(processed_feat)

        return self.refined_conv(self.sum(torch.stack(processed_features), dim=0))


class C2PSA(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(C2PSA, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_reduce = nn.Linear(in_channels,
                                   in_channels // reduction_ratio,
                                   bias=False
                                   )
        self.fc_expand = nn.Linear(in_channels // reduction_ratio,
                                   in_channels,
                                   bias=False
                                   )
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        # Convolutional layer to generate pixel-wise attention weights
        self.conv_pixel_attention = nn.Conv2d(1,
                                              1, kernel_size=3,
                                              padding=1,
                                              bias=True)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Global Average Pooling: shape [batch_size, channels]
        pooled = self.global_avg_pool(x).view(batch_size, channels)

        # Fully connected layers to reduce and expand dimensions
        reduced = self.fc_reduce(pooled)
        activated = self.relu(reduced)
        expanded = self.fc_expand(activated)

        # reshape to [batch-size, channels, 1, 1] and apply activation for attention weights
        channel_attention_weights = torch.sigmoid(self.conv_pixel_attention(expanded).unsqueeze(-1).unsqueeze(-1))

        # Apply channel_wise attention
        attended_channels = x * channel_attention_weights

        # generate pixel_wise attention
        global_avg_channel_features = attended_channels.mean(dim=1, keepdim=True)
        pixel_attention_weights = torch.sigmoid(self.conv_pixel_attention(global_avg_channel_features))

        # Return the result of pixel_wise attention
        return attended_channels * pixel_attention_weights


class Yolov11(nn.Module):
    def __init__(self, num_classes=20, in_channels=3, **kwargs):
        super(Yolov11, self).__init__()
        self.back_bone_architecture = back_bone_architecture
        self.neck_architecture = neck_architecture

        self.num_classes = num_classes
        self.in_channels = in_channels

        # A diction that contains the tensors that are exchanged between different blocks
        self.communication = {}

        # Calculate the feature dimension
        x = torch.randn((1, in_channels, 640, 640))
        with torch.no_grad():
            x = self.net(x)
        self.feature_size = x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x):
        x = self._partly_forward('back_bone', x)
        x = self._partly_forward('neck', x)

        return x


    def _partly_forward(self, config_name, input_tensor):
        if config_name == 'back_bone':
            config = self.back_bone_architecture
        elif config_name == 'neck':
            config = self.neck_architecture
        else: raise ValueError('Invalid config name')

        counter = 0
        in_channels = self.in_channels
        for x in config:
            op_name = x[0]
            op_params = x[1]
            repeat_num = x[-1]

            for i in range(repeat_num):
                if (op_name == 'Conv') | (op_name == 'C3K2'):
                    layer = CNNblock(in_channels=in_channels,
                                out_channels=op_params[1],
                                kernel_size=op_params[2],
                                stride=op_params[3],
                                padding=op_params[4],
                                )
                    output = layer(input_tensor)

                    self.communication[counter] = output
                    in_channels = x[1]
                    input_tensor = output
                    counter += 1
