'''
This is an adaptaion of https://github.com/xsyl0011/SAIPP-Net

@inproceedings{Feng2025SAIPPNet,
  title={SAIPP-Net: A Sampling-Assisted Indoor Pathloss Prediction Method for Wireless Communication Systems},
  author={Feng, Bin and Zheng, Meng and Liang, Wei and Zhang, Lei},
  booktitle={Proc. IEEE 35th International Workshop on Machine Learning for Signal Processing (MLSP)},
  year={2025},
  month={August}
}

MIT License

Copyright (c) 2025 FENG BIN

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

def convrelu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True), 
    )

maxpool = nn.MaxPool2d(2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)

class SDUblock(nn.Module):
    def __init__(self, in_channels, n_out):
        super(SDUblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, n_out // 2, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(n_out // 2, n_out // 4, kernel_size=3, padding=3, dilation=3)
        self.conv3 = nn.Conv2d(n_out // 4, n_out // 8, kernel_size=3, padding=6, dilation=6)
        self.conv4 = nn.Conv2d(n_out // 8, n_out // 16, kernel_size=3, padding=9, dilation=9)
        self.conv5 = nn.Conv2d(n_out // 16, n_out // 16, kernel_size=3, padding=12, dilation=12)
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(out1))
        out3 = self.relu(self.conv3(out2))
        out4 = self.relu(self.conv4(out3))
        out5 = self.relu(self.conv5(out4))
        return torch.cat([out1, out2, out3, out4, out5], dim=1)

def convreluT(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
        nn.ReLU(inplace=True),
    )

def upsample(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
    )

class RadioNetAnySize(nn.Module):
    def __init__(self, inputs=6, initial_downsampling=1, *args, **kwargs) -> None:
        super().__init__()
        if initial_downsampling==2:
            self.model = RadioNet64to32(inputs)
        elif initial_downsampling==4:
            self.model = RadioNet128to32(inputs)
        elif initial_downsampling==8:
            self.model = RadioNet256to32(inputs)
        else:
            raise ValueError(f'{initial_downsampling=}')
    
    def forward(self, input, z=None):
        return self.model(input, z)


class RadioNet64to32(nn.Module):
    def __init__(self, inputs=6):
        super().__init__()
        self.inputs = inputs
        
        # Encoder - designed for 64x64 input
        self.encode1_conv = convrelu(inputs, 32)       # 64x64 -> 64x64
        self.encode1_sdu = SDUblock(32, 32)
        self.maxpool = maxpool                          # 64x64 -> 32x32
        
        self.encode2_conv = convrelu(32, 64)           # 32x32 -> 32x32  
        self.encode2_sdu = SDUblock(64, 64)
        # maxpool                                       # 32x32 -> 16x16
        
        self.encode3_conv = convrelu(64, 128)          # 16x16 -> 16x16
        self.encode3_sdu = SDUblock(128, 128) 
        # maxpool                                       # 16x16 -> 8x8
        
        self.encode4_conv = convrelu(128, 256)         # 8x8 -> 8x8
        self.encode4_sdu = SDUblock(256, 256)
        # maxpool                                       # 8x8 -> 4x4
        
        # Bottleneck
        self.bottleneck_conv = convrelu(256, 512)      # 4x4 -> 4x4
        self.bottleneck_sdu = SDUblock(512, 512)
        
        # Decoder - from 4x4 to 32x32 (only 2 upsampling stages needed)
        self.upconv1 = convreluT(512, 256)            # 4x4 -> 8x8
        self.decode1_conv = convrelu(256*2, 256)      # Skip connection from encode4
        self.decode1_sdu = SDUblock(256, 256)
        
        self.upconv2 = convreluT(256, 128)            # 8x8 -> 16x16  
        self.decode2_conv = convrelu(128*2, 128)      # Skip connection from encode3
        self.decode2_sdu = SDUblock(128, 128)
        
        self.upconv3 = convreluT(128, 64)             # 16x16 -> 32x32
        self.decode3_conv = convrelu(64*2, 64)        # Skip connection from encode2
        self.decode3_sdu = SDUblock(64, 64)
        
        # Final output layer
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)  # 32x32 -> 32x32
        
        # Keep your learnable parameters
        self.delta_1 = nn.Parameter(torch.tensor(1.0))
        self.delta_2 = nn.Parameter(torch.tensor(1.0))
        self.delta_3 = nn.Parameter(torch.tensor(1.0))
        self.delta_4 = nn.Parameter(torch.tensor(1.0))
        self.delta_6 = nn.Parameter(torch.tensor(1.0))
        self.delta_10 = nn.Parameter(torch.tensor(1.0))
        self.delta_23 = nn.Parameter(torch.tensor(1.0))
        self.material_weights = nn.Parameter(torch.ones(7))

    def forward(self, input, z=None):
        input0 = input
        
        # Encoder path
        encode1_conv = self.encode1_conv(input0)       # 64x64
        encode1_sdu = self.encode1_sdu(encode1_conv)
        encode1_pool = self.maxpool(encode1_sdu)       # 32x32
        
        encode2_conv = self.encode2_conv(encode1_pool)  # 32x32
        encode2_sdu = self.encode2_sdu(encode2_conv)
        encode2_pool = self.maxpool(encode2_sdu)       # 16x16
        
        encode3_conv = self.encode3_conv(encode2_pool)  # 16x16
        encode3_sdu = self.encode3_sdu(encode3_conv)
        encode3_pool = self.maxpool(encode3_sdu)       # 8x8
        
        encode4_conv = self.encode4_conv(encode3_pool)  # 8x8
        encode4_sdu = self.encode4_sdu(encode4_conv)
        encode4_pool = self.maxpool(encode4_sdu)       # 4x4
        
        # Bottleneck
        bottleneck_conv = self.bottleneck_conv(encode4_pool)  # 4x4
        bottleneck_sdu = self.bottleneck_sdu(bottleneck_conv)
        
        # Decoder path
        decode1_up = self.upconv1(bottleneck_sdu)      # 4x4 -> 8x8
        decode1_cat = torch.cat([decode1_up, encode4_sdu], dim=1)
        decode1_conv = self.decode1_conv(decode1_cat)
        decode1_sdu = self.decode1_sdu(decode1_conv)
        
        decode2_up = self.upconv2(decode1_sdu)         # 8x8 -> 16x16
        decode2_cat = torch.cat([decode2_up, encode3_sdu], dim=1)
        decode2_conv = self.decode2_conv(decode2_cat)
        decode2_sdu = self.decode2_sdu(decode2_conv)
        
        decode3_up = self.upconv3(decode2_sdu)         # 16x16 -> 32x32
        decode3_cat = torch.cat([decode3_up, encode2_sdu], dim=1)
        decode3_conv = self.decode3_conv(decode3_cat)
        decode3_sdu = self.decode3_sdu(decode3_conv)
        
        output = self.output(decode3_sdu)              # 32x32
        
        return output

    
    
class RadioNet128to32(nn.Module):
    def __init__(self, inputs=6):
        super().__init__()
        self.inputs = inputs
        
        # Encoder - designed for 128x128 input
        self.encode1_conv = convrelu(inputs, 32)       # 128x128 -> 128x128
        self.encode1_sdu = SDUblock(32, 32)
        self.maxpool = maxpool                          # 128x128 -> 64x64
        
        self.encode2_conv = convrelu(32, 64)           # 64x64 -> 64x64  
        self.encode2_sdu = SDUblock(64, 64)
        # maxpool                                       # 64x64 -> 32x32
        
        self.encode3_conv = convrelu(64, 128)          # 32x32 -> 32x32
        self.encode3_sdu = SDUblock(128, 128) 
        # maxpool                                       # 32x32 -> 16x16
        
        self.encode4_conv = convrelu(128, 256)         # 16x16 -> 16x16
        self.encode4_sdu = SDUblock(256, 256)
        # maxpool                                       # 16x16 -> 8x8
        
        self.encode5_conv = convrelu(256, 512)         # 8x8 -> 8x8
        self.encode5_sdu = SDUblock(512, 512)
        # maxpool                                       # 8x8 -> 4x4
        
        # Bottleneck
        self.bottleneck_conv = convrelu(512, 1024)     # 4x4 -> 4x4
        self.bottleneck_sdu = SDUblock(1024, 1024)
        
        # Decoder - from 4x4 to 32x32 (only 3 upsampling stages needed)
        self.upconv1 = convreluT(1024, 512)           # 4x4 -> 8x8
        self.decode1_conv = convrelu(512*2, 512)      # Skip connection from encode5
        self.decode1_sdu = SDUblock(512, 512)
        
        self.upconv2 = convreluT(512, 256)            # 8x8 -> 16x16  
        self.decode2_conv = convrelu(256*2, 256)      # Skip connection from encode4
        self.decode2_sdu = SDUblock(256, 256)
        
        self.upconv3 = convreluT(256, 128)            # 16x16 -> 32x32
        self.decode3_conv = convrelu(128*2, 128)      # Skip connection from encode3
        self.decode3_sdu = SDUblock(128, 128)
        
        # Final output layer
        self.output = nn.Conv2d(128, 1, kernel_size=1, padding=0)  # 32x32 -> 32x32
        
        # Keep your learnable parameters
        self.delta_1 = nn.Parameter(torch.tensor(1.0))
        self.delta_2 = nn.Parameter(torch.tensor(1.0))
        self.delta_3 = nn.Parameter(torch.tensor(1.0))
        self.delta_4 = nn.Parameter(torch.tensor(1.0))
        self.delta_6 = nn.Parameter(torch.tensor(1.0))
        self.delta_10 = nn.Parameter(torch.tensor(1.0))
        self.delta_23 = nn.Parameter(torch.tensor(1.0))
        self.material_weights = nn.Parameter(torch.ones(7))

    def forward(self, input, z=None):
        input0 = input
        
        # Encoder path
        encode1_conv = self.encode1_conv(input0)       # 128x128
        encode1_sdu = self.encode1_sdu(encode1_conv)
        encode1_pool = self.maxpool(encode1_sdu)       # 64x64
        
        encode2_conv = self.encode2_conv(encode1_pool)  # 64x64
        encode2_sdu = self.encode2_sdu(encode2_conv)
        encode2_pool = self.maxpool(encode2_sdu)       # 32x32
        
        encode3_conv = self.encode3_conv(encode2_pool)  # 32x32
        encode3_sdu = self.encode3_sdu(encode3_conv)
        encode3_pool = self.maxpool(encode3_sdu)       # 16x16
        
        encode4_conv = self.encode4_conv(encode3_pool)  # 16x16
        encode4_sdu = self.encode4_sdu(encode4_conv)
        encode4_pool = self.maxpool(encode4_sdu)       # 8x8
        
        encode5_conv = self.encode5_conv(encode4_pool)  # 8x8
        encode5_sdu = self.encode5_sdu(encode5_conv)
        encode5_pool = self.maxpool(encode5_sdu)       # 4x4
        
        # Bottleneck
        bottleneck_conv = self.bottleneck_conv(encode5_pool)  # 4x4
        bottleneck_sdu = self.bottleneck_sdu(bottleneck_conv)
        
        # Decoder path
        decode1_up = self.upconv1(bottleneck_sdu)      # 4x4 -> 8x8
        decode1_cat = torch.cat([decode1_up, encode5_sdu], dim=1)
        decode1_conv = self.decode1_conv(decode1_cat)
        decode1_sdu = self.decode1_sdu(decode1_conv)
        
        decode2_up = self.upconv2(decode1_sdu)         # 8x8 -> 16x16
        decode2_cat = torch.cat([decode2_up, encode4_sdu], dim=1)
        decode2_conv = self.decode2_conv(decode2_cat)
        decode2_sdu = self.decode2_sdu(decode2_conv)
        
        decode3_up = self.upconv3(decode2_sdu)         # 16x16 -> 32x32
        decode3_cat = torch.cat([decode3_up, encode3_sdu], dim=1)
        decode3_conv = self.decode3_conv(decode3_cat)
        decode3_sdu = self.decode3_sdu(decode3_conv)
        
        output = self.output(decode3_sdu)              # 32x32
        
        return output

    
class RadioNet256to32(nn.Module):
    def __init__(self, inputs=6):
        super().__init__()
        self.inputs = inputs
        
        # Encoder - designed for 256x256 input
        self.encode1_conv = convrelu(inputs, 32)       # 256x256 -> 256x256
        self.encode1_sdu = SDUblock(32, 32)
        self.maxpool = maxpool                          # 256x256 -> 128x128
        
        self.encode2_conv = convrelu(32, 64)           # 128x128 -> 128x128  
        self.encode2_sdu = SDUblock(64, 64)
        # maxpool                                       # 128x128 -> 64x64
        
        self.encode3_conv = convrelu(64, 128)          # 64x64 -> 64x64
        self.encode3_sdu = SDUblock(128, 128) 
        # maxpool                                       # 64x64 -> 32x32
        
        self.encode4_conv = convrelu(128, 256)         # 32x32 -> 32x32
        self.encode4_sdu = SDUblock(256, 256)
        # maxpool                                       # 32x32 -> 16x16
        
        self.encode5_conv = convrelu(256, 512)         # 16x16 -> 16x16
        self.encode5_sdu = SDUblock(512, 512)
        # maxpool                                       # 16x16 -> 8x8
        
        self.encode6_conv = convrelu(512, 512)         # 8x8 -> 8x8
        self.encode6_sdu = SDUblock(512, 512)
        # maxpool                                       # 8x8 -> 4x4
        
        # Bottleneck
        self.bottleneck_conv = convrelu(512, 1024)     # 4x4 -> 4x4
        self.bottleneck_sdu = SDUblock(1024, 1024)
        
        # Decoder - from 4x4 to 32x32 (4 upsampling stages needed)
        self.upconv1 = convreluT(1024, 512)           # 4x4 -> 8x8
        self.decode1_conv = convrelu(512*2, 512)      # Skip connection from encode6
        self.decode1_sdu = SDUblock(512, 512)
        
        self.upconv2 = convreluT(512, 512)            # 8x8 -> 16x16  
        self.decode2_conv = convrelu(512*2, 512)      # Skip connection from encode5
        self.decode2_sdu = SDUblock(512, 512)
        
        self.upconv3 = convreluT(512, 256)            # 16x16 -> 32x32
        self.decode3_conv = convrelu(256*2, 256)      # Skip connection from encode4
        self.decode3_sdu = SDUblock(256, 256)
        
        # Final output layer
        self.output = nn.Conv2d(256, 1, kernel_size=1, padding=0)  # 32x32 -> 32x32
        
        # Keep your learnable parameters
        self.delta_1 = nn.Parameter(torch.tensor(1.0))
        self.delta_2 = nn.Parameter(torch.tensor(1.0))
        self.delta_3 = nn.Parameter(torch.tensor(1.0))
        self.delta_4 = nn.Parameter(torch.tensor(1.0))
        self.delta_6 = nn.Parameter(torch.tensor(1.0))
        self.delta_10 = nn.Parameter(torch.tensor(1.0))
        self.delta_23 = nn.Parameter(torch.tensor(1.0))
        self.material_weights = nn.Parameter(torch.ones(7))

    def forward(self, input, z=None):
        input0 = input
        
        # Encoder path
        encode1_conv = self.encode1_conv(input0)       # 256x256
        encode1_sdu = self.encode1_sdu(encode1_conv)
        encode1_pool = self.maxpool(encode1_sdu)       # 128x128
        
        encode2_conv = self.encode2_conv(encode1_pool)  # 128x128
        encode2_sdu = self.encode2_sdu(encode2_conv)
        encode2_pool = self.maxpool(encode2_sdu)       # 64x64
        
        encode3_conv = self.encode3_conv(encode2_pool)  # 64x64
        encode3_sdu = self.encode3_sdu(encode3_conv)
        encode3_pool = self.maxpool(encode3_sdu)       # 32x32
        
        encode4_conv = self.encode4_conv(encode3_pool)  # 32x32
        encode4_sdu = self.encode4_sdu(encode4_conv)
        encode4_pool = self.maxpool(encode4_sdu)       # 16x16
        
        encode5_conv = self.encode5_conv(encode4_pool)  # 16x16
        encode5_sdu = self.encode5_sdu(encode5_conv)
        encode5_pool = self.maxpool(encode5_sdu)       # 8x8
        
        encode6_conv = self.encode6_conv(encode5_pool)  # 8x8
        encode6_sdu = self.encode6_sdu(encode6_conv)
        encode6_pool = self.maxpool(encode6_sdu)       # 4x4
        
        # Bottleneck
        bottleneck_conv = self.bottleneck_conv(encode6_pool)  # 4x4
        bottleneck_sdu = self.bottleneck_sdu(bottleneck_conv)
        
        # Decoder path
        decode1_up = self.upconv1(bottleneck_sdu)      # 4x4 -> 8x8
        decode1_cat = torch.cat([decode1_up, encode6_sdu], dim=1)
        decode1_conv = self.decode1_conv(decode1_cat)
        decode1_sdu = self.decode1_sdu(decode1_conv)
        
        decode2_up = self.upconv2(decode1_sdu)         # 8x8 -> 16x16
        decode2_cat = torch.cat([decode2_up, encode5_sdu], dim=1)
        decode2_conv = self.decode2_conv(decode2_cat)
        decode2_sdu = self.decode2_sdu(decode2_conv)
        
        decode3_up = self.upconv3(decode2_sdu)         # 16x16 -> 32x32
        decode3_cat = torch.cat([decode3_up, encode4_sdu], dim=1)
        decode3_conv = self.decode3_conv(decode3_cat)
        decode3_sdu = self.decode3_sdu(decode3_conv)
        
        output = self.output(decode3_sdu)              # 32x32
        
        return output
