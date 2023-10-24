import torch.nn as nn
import govars
import torch

class Baseline_CNN(nn.Module):
    def __init__(self, num_layers=8, hidden_dim=128):
        # in the current version the shallower network perform better, need residual connections maybe?
        super().__init__()
        
        self.net = nn.ModuleList([
            nn.Conv2d(govars.FEAT_CHNLS, hidden_dim, kernel_size=7, stride=1, padding=3), 
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2), 
            nn.LeakyReLU()
            ])

        for _ in range(num_layers - 1):
            self.net.extend([nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1), nn.LeakyReLU()])

        self.fc = nn.Linear(hidden_dim * govars.SIZE * govars.SIZE, govars.ACTION_SPACE)
        self.drop_out = nn.Dropout()
        

    def forward(self, x):
        for module in self.net:
            x = module(x) 
        x = torch.flatten(x, start_dim=1)
        x = self.drop_out(self.fc(x))
        return x





class ResNet(nn.Module):
    def __init__(self, num_layers=8, hidden_dim=128):
        # in the current version the shallower network perform better, need residual connections maybe?
        super().__init__()
        
        self.net = nn.ModuleList([
            nn.Conv2d(govars.FEAT_CHNLS, hidden_dim, kernel_size=7, stride=1, padding=3), 
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2), 
            nn.LeakyReLU()
            ])

        for _ in range(num_layers - 1):
            self.net.extend([ResBlock(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)])

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * govars.PADDED_SIZE * govars.PADDED_SIZE, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 3)
        )
        self.drop_out = nn.Dropout(p=0.2)
        

    def forward(self, x):
        for module in self.net:
            x = module(x) 
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()


        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(),
        )

        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        identity = x
        cnn_result = self.cnn(x)

        output = identity + cnn_result
        output = self.relu(output)
        return output


def get_model(model_type):
    if model_type == 'basic':   
        return Baseline_CNN()
    if model_type == 'resnet':
        return ResNet()
    