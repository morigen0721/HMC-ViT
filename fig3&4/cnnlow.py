from torch.utils.data import TensorDataset
import torch
import numpy as np
from torch import nn
from torch import optim

class lowsnr_cnn(nn.Module):
    def __init__(self):
        super(lowsnr_cnn, self).__init__()                 #输入为batch*3*16*16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=4096, out_features=4096)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)

        self.linear2 = nn.Linear(in_features=4096,out_features=2048)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)

        self.linear3 = nn.Linear(in_features=2048,out_features=1024)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.3)

        self.linear4 = nn.Linear(in_features=1024, out_features=121)
        self.sigmod = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)

        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.linear4(x)
        return x

# if __name__ == "__main__":
#     net = lowsnr_cnn()
#     print(net)
#     summary(net, input_size=(100, 3, 16, 16) )

