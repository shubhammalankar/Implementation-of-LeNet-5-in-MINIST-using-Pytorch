import torch.nn as nn
import torch.nn.functional as functional

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.tanH = nn.Tanh()
        self.averagePool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.convolutionLayer1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1),
                                           padding=(0, 0))
        self.convolutionLayer2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1),
                                           padding=(0, 0))
        self.convolutionLayer3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1),
                                           padding=(0, 0))
        self.linearLayer1 = nn.Linear(120, 84)
        self.linearLayer2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.tanH(self.convolutionLayer1(x))
        x = self.averagePool(x)
        x = self.tanH(self.convolutionLayer2(x))
        x = self.averagePool(x)

        x = self.tanH(self.convolutionLayer3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.tanH(self.linearLayer1(x))
        x = self.linearLayer2(x)
        probs = functional.softmax(x, dim=1)
        return x