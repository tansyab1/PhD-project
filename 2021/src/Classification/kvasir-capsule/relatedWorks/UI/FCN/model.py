import torch
import torch.nn as nn

# create a vanilla UNet model
class FCN(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        
        self.conv1 = bottleneck(in_channels, out_channels)
        self.conv2 = bottleneck(out_channels, out_channels)
        self.conv3 = bottleneck(out_channels, out_channels)
        
        self.deconv1 = deconv(out_channels, out_channels)
        self.deconv2 = deconv(out_channels, out_channels)
        self.deconv3 = deconv(out_channels, out_channels)
        
        self.res = residual(in_channels=67, out_channels=128)
        
    
    def forward(self, x):
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(conv1_x)
        conv3_x = self.conv3(conv2_x)
        
        deconv1_x = self.deconv1(conv3_x)
        deconv2_x = self.deconv2(deconv1_x+conv2_x)
        deconv3_x = self.deconv3(deconv2_x+conv1_x)
        out = self.res(torch.concat((deconv3_x, x), 1))
        
        return out
    
    
class bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        conv_x = self.conv(x)
        return conv_x
        
        
class deconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.Sequential(
            # upsampling with bilinear interpolation
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, x):
        deconv_x = self.deconv(x)
        return deconv_x
    


class residual(nn.Module):
    def __init__(self, in_channels = 67, out_channels=128):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(out_channels, 3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        conv_x = self.conv(x)
        conv_x_2 = self.conv2(conv_x)
        conv_x_3 = self.conv3(conv_x_2+conv_x)
        conv_x_4 = self.conv4(conv_x_3+conv_x_2)
        out = self.conv5(conv_x_4)
        
        return out