import numpy as np
from numpy.lib.arraypad import pad
import torch
from torch.autograd.variable import Variable
import torch.nn as nn
import os 
# from torchsummary import summary


class Encoder_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ultra_stream = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2, 5), dilation=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 5), dilation=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 5), dilation=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 5), dilation=(1, 5)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=(2, 5), dilation=(1, 1)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 1)),
        )
        self.audio_stream = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), dilation=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), dilation=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), dilation=(5, 5)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 1)),
        )
    def forward(self, ultra, audio):
        ultra_out = self.ultra_stream(ultra)
        audio_out = self.audio_stream(audio)
        out = torch.cat([ultra_out, audio_out], dim=2)
        return out




class Residual_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(4)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(4)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(4)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(4)
        self.relu5 = nn.ReLU()
    def forward(self, x):
        # x = torch.reshape(x, (-1, 128, 13))

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)
        
        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out2 = self.relu2(out1+out2)

        out3 = self.conv3(out2)
        out3 = self.bn3(out3)
        out3 = self.relu3(out2+out3)

        out4 = self.conv4(out3)
        out4 = self.bn4(out4)
        out4 = self.relu4(out3+out4)

        out5 = self.conv5(out4)
        out5 = self.bn5(out5)
        out5 = self.relu5(out5)

        # out = torch.reshape(out5, (-1, 4, 32, 13)) 
        return out5
    


class Decoder_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=(5, 5), stride=(1, 1)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=(4, 5), stride=(1, 2), dilation=(5, 5)),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=(4, 6), stride=(1, 2), dilation=(3, 3)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(5, 6), stride=(2, 2), dilation=(2, 2)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(5, 5), stride=(2, 2)),
            nn.BatchNorm2d(1),
            # nn.ReLU(),
            nn.Sigmoid()
    
        )
    def forward(self, x):
        out = self.upsample(x)
        return out
 


class Dricriminator1(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fcn = nn.Sequential(
            nn.Linear(4*32*13, 600),
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Linear(300, 150),
            nn.ReLU(),
            # nn.Linear(150, 50),
            # nn.ReLU(),
            nn.Linear(150, 2),
            # nn.ReLU(),
            nn.Softmax(dim=1)
            # nn.Sigmoid()
        )
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        out = self.fcn(x)
        
        return out
   


class Dricriminator2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcn = nn.Sequential(
            nn.Linear(4*32*13, 600),
            nn.ReLU(),
            nn.Linear(600, 50),
            # nn.ReLU(),
            # nn.Softmax(dim=1)
            # nn.Sigmoid()
            nn.Sigmoid()
        )
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        out = self.fcn(x)
        return out

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Encoder_Block().to(device)
# model1 = Residual_Block().to(device)
# model2 = Decoder_Block().to(device)
# model3 = Dricriminator1().to(device)
# # summary(model, [(8, 16, 501), (1, 257, 501)])
# summary(model1, (4, 32, 13))