import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class PerceptionLoss(nn.Module):
    def __init__(self):
        super(PerceptionLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X, Y):
        h_x = X
        h_y = Y
        h1_x = self.slice1(h_x)
        h1_y = self.slice1(h_y)
        h2_x = self.slice2(h1_x)
        h2_y = self.slice2(h1_y)
        h3_x = self.slice3(h2_x)
        h3_y = self.slice3(h2_y)
        h4_x = self.slice4(h3_x)
        h4_y = self.slice4(h3_y)
        h5_x = self.slice5(h4_x)
        h5_y = self.slice5(h4_y)
        loss = (
            F.mse_loss(h1_x, h1_y)
            + F.mse_loss(h2_x, h2_y)
            + F.mse_loss(h3_x, h3_y)
            + F.mse_loss(h4_x, h4_y)
            + F.mse_loss(h5_x, h5_y)
        )
        return loss
