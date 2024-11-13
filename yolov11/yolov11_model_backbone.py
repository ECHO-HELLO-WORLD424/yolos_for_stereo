
import torch
import torch.nn as nn



'''Global Definition Of Repeat factor'''
R = 3
'''Input image: RGB, 640 * 640'''
# Output size of conv = [(Input size + 2×Padding - Kernel size) / Stride] + 1
# Output size of max pool = [(Input size - Kernel size) / Stride] + 1
back_bone_architecture = {
    # CNN layer tuple (input_channel, out_channels, kernel, stride, padding)
    # CNN layer list with repeat num ['operation_name', (in, out, k, s, p), repeat_num]
    ['Conv', (3, 64, 3, 2, 1), 1],
    ['Conv', (64, 128, 3, 2, 1), 1],
    ['conv', (128, 128, 3, 2, 1), 3 * R],
    ['Conv', (128, 256, 3, 2, 1), 1],
    ['Conv',(256, 256, 3, 2, 1), 3 * R],
    ['Conv', (256, 512, 3, 2, 1), 1],
    ['Conv', (512, 512, 3, 2, 1), 3 * R],
    ['Conv', (512, 1024, 3, 2, 1), 1]
}

neck_architecture = {
    ['SPFF'],
    ['C2PSA'],
    # Up sample list: ['operation_name', output_size(D, H, W), scale_factor]
    ['Upsample', (512, 40, 40)],
    # Concat list: ['operation_name', dimension, (tensor1, tensor2)]
    ["Concat", 0, ('back_bone_c3k2_3', 'neck_Upsample_1')],
    ['Conv', (512, 512, 3, 2, 1), 3 * R],
    ['Upsample', (256, 80, 80)],
    ["Concat", 0, ('back_bone_c3k2_3', 'neck_Upsample_2')],
    ['Conv', (256, 256, 3, 2, 1), 3 * R],
}