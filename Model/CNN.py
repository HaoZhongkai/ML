from .Basic_Model import Basic_Model
import torch as T
import torch.nn as nn
class CNN_Cifar(Basic_Model):
    def __init__(self):
        super(CNN_Cifar,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=30,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                30,40,5,1,2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                40,80,4,2,1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.linear = nn.Linear(80*2*2,100)
        self.out = nn.Linear(100,10)

    def forward(self,x):
        x = x.view(-1,3,32,32)
        x = self.conv1(x)               #10*16*16
        x = self.conv2(x)               #20*8*8
        x = self.conv3(x)               # 2*2
        x = x.view(x.size(0),-1)
        x = self.linear(x)              #100
        x = self.out(x)                 #10
        return x