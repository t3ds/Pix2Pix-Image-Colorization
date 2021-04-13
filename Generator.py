import torch
import torch.nn as nn
from utils import ConvBlock, DecodeBlock

class Generator(nn.Module):
    def __init__(self, in_channels, hidden_channels = 64, out_channels = 2):
        super(Generator, self).__init__()

        self.initial = ConvBlock(in_channels, hidden_channels)
        self.contract1 = ConvBlock(hidden_channels, hidden_channels*2, use_bn=True, use_dropout=False)
        self.contract2 = ConvBlock(hidden_channels*2, hidden_channels*4, use_bn=True, use_dropout=False)
        self.contract3 = ConvBlock(hidden_channels*4, hidden_channels*8, use_bn=True, use_dropout=False)
        self.contract4 = ConvBlock(hidden_channels*8, hidden_channels*8, use_bn=True, use_dropout=False)
        self.contract5 = ConvBlock(hidden_channels*8, hidden_channels*8, use_bn=True, use_dropout=False)
        self.contract6= ConvBlock(hidden_channels*8, hidden_channels*8, use_bn=True, use_dropout=False)
        self.contract7 = ConvBlock(hidden_channels*8, hidden_channels*8, use_bn=False, use_dropout=False)

        hidden_channels = hidden_channels*8
        self.expand1 = DecodeBlock(hidden_channels, hidden_channels, use_dropout=True, use_bn=True)
        self.expand2 = DecodeBlock(hidden_channels*2, hidden_channels, use_dropout=True, use_bn=True)
        self.expand3 = DecodeBlock(hidden_channels*2, hidden_channels, use_dropout=True, use_bn=True)
        self.expand4 = DecodeBlock(hidden_channels*2, hidden_channels, use_dropout=False, use_bn=True)
        self.expand5 = DecodeBlock(hidden_channels*2, hidden_channels//2, use_dropout=False, use_bn=True)
        self.expand6 = DecodeBlock(hidden_channels, hidden_channels//4, use_dropout=False, use_bn=True)
        self.expand7 = DecodeBlock(hidden_channels//2, hidden_channels//8, use_dropout=False, use_bn=True)
        #self.final_up = DecodeBlock(hidden_channels//4, hidden_channels//8, use_dropout=False, use_bn=False, stride  = 2)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels//4, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
    

    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.contract5(x4)
        x6 = self.contract6(x5)
        x7 = self.contract7(x6)
        x8 = self.expand1(x7)
        x9 = self.expand2(torch.cat([x8, x6], axis= 1))
        x10 = self.expand3(torch.cat([x9, x5], axis= 1))
        x11 = self.expand4(torch.cat([x10, x4], axis= 1))
        x12 = self.expand5(torch.cat([x11, x3], axis= 1))
        x13 = self.expand6(torch.cat([x12, x2], axis= 1))
        x14 = self.expand7(torch.cat([x13, x1], axis= 1))
        #x15 = self.final_up(torch.cat([x14, x0], axis= 1))
        return self.final(torch.cat([x14, x0], axis= 1))

if __name__ == "__main__":
    def test_gen():
        x = torch.randn((2,1,256,256))

        model = Generator(1, out_channels=2)
        pred = model(x)
        print(pred.shape)

    test_gen()