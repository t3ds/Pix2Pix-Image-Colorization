import torch
import torch.nn as nn
from utils import ConvBlock

class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, hidden_channels = 64):
        super(Discriminator, self).__init__()

        self.initial = ConvBlock(in_channels, hidden_channels)

        self.block1 = ConvBlock(hidden_channels, hidden_channels*2, use_bn= True) #64 -> 128
        self.block2 = ConvBlock(hidden_channels*2, hidden_channels*4, use_bn= True) #128 -> 256
        self.block3 = ConvBlock(hidden_channels*4, hidden_channels*8, use_bn= True) #256 -> 512
        self.block4 = ConvBlock(hidden_channels*8, hidden_channels*8, use_bn= True, stride = 1) #512
        #self.block5 = ConvBlock(hidden_channels*8, hidden_channels*8, use_bn= True, stride = 1) 
        #512
        self.final = nn.Conv2d(hidden_channels * 8, 1, kernel_size=1, padding= 1, padding_mode="reflect")
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.initial(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        #x5 = self.block5(x4)
        x6 = self.final(x4)
        xn = self.sigmoid(x6)
        return xn

if __name__ == "__main__":
    def test():
        x = torch.randn((2,3,256,256))
        y = torch.randn((2,3,256,256))

        model = Discriminator(6)
        pred = model(x,y)
        print(pred.shape)

    test()