import torch.nn as nn
import govars
import torch


class ResNet(nn.Module):
    '''
    resnet without any downsampling, and the input is padded to ensure the output's size is the same as input.
    '''
    def __init__(self, num_blocks=13, hidden_dim=64):


        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=govars.FEAT_CHNLS, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
        )

        self.res_blocks = nn.ModuleList([ResBlock(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1
        ) for _ in range(num_blocks)])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_dim, 3)


    def forward(self, x):
        x = self.head(x)
        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

        


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.net(x)

        out = out + identity

        return self.relu(out)
