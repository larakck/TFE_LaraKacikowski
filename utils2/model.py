import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(4, out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(4, out_ch),
                nn.ReLU(),
                nn.Dropout(0.3)
            )

        self.enc1 = CBR(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = CBR(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = CBR(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = CBR(512, 1024)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = nn.Conv2d(1024, 512, kernel_size=1)
        self.dec4 = CBR(1024, 512)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=1)
        self.dec3 = CBR(512, 256)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1)
        self.dec2 = CBR(256, 128)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(128, 64, kernel_size=1)
        self.dec1 = CBR(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        up4 = self.conv4(self.up4(bottleneck))
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))

        up3 = self.conv3(self.up3(dec4))
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))

        up2 = self.conv2(self.up2(dec3))
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = self.conv1(self.up1(dec2))
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        return self.final(dec1)

