import torch
from torch import nn
from torch import optim

from unet import Unet
from discriminator import MultiScaleDiscriminator
from ganloss import GANLoss
from utils import lab_to_rgb


class MainModel(nn.Module):
    def __init__(
        self,
        net_G=None,
        lr_G=1e-4,
        lr_D=4e-4,
        beta1=0.5,
        beta2=0.999,
        lambda_L1=100.0,
        lambda_GAN=1.0,
        gan_mode="vanilla"
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        self.lambda_GAN = lambda_GAN

        if net_G is None:
            self.net_G = Unet(input_c=1, output_c=2, n_down=8, num_filters=64).to(self.device)
        else:
            self.net_G = net_G.to(self.device)

        self.net_D = MultiScaleDiscriminator(input_c=3, num_filters=64, n_layers=5, num_D=3).to(self.device)
        self.GANcriterion = GANLoss(gan_mode=gan_mode).to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

        self.loss_G_GAN = None
        self.loss_G_L1 = None
        self.loss_D = None
        self.loss_D_real = None
        self.loss_D_fake = None

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, l_chan, ab_chan):
        self.l_chan = l_chan.to(self.device)
        self.ab_chan = ab_chan.to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.l_chan)

    def backward_D(self):
        fake_image = torch.cat([self.l_chan, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.l_chan, self.ab_chan], dim=1)
        real_preds = self.net_D(real_image)
        loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (loss_D_fake + loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.l_chan, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True) * self.lambda_GAN
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab_chan) * self.lambda_L1
        
        real_rgb = lab_to_rgb(self.l_chan, self.ab_chan)
        fake_rgb = lab_to_rgb(self.l_chan, self.fake_color)
        
        real_rgb = real_rgb.permute(0, 3, 1, 2)
        fake_rgb = fake_rgb.permute(0, 3, 1, 2)
        
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        self.forward()

        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()
    
    def compute_D_loss(self):
        real_image = torch.cat([self.l_chan, self.ab_chan], dim=1)
        fake_image = torch.cat([self.l_chan, self.fake_color.detach()], dim=1)
        
        real_preds = self.net_D(real_image)
        fake_preds = self.net_D(fake_image)
        
        self.loss_D_real = 0
        self.loss_D_fake = 0
        for real_scale_preds, fake_scale_preds in zip(real_preds, fake_preds):
            self.loss_D_real += self.GANcriterion(real_scale_preds[-1], True)
            self.loss_D_fake += self.GANcriterion(fake_scale_preds[-1], False)
        
        self.loss_D_real /= len(real_preds)
        self.loss_D_fake /= len(fake_preds)
        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        
        return self.loss_D

    def compute_G_loss(self):
        fake_image = torch.cat([self.l_chan, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        
        self.loss_G_GAN = 0
        for scale_preds in fake_preds:
            self.loss_G_GAN += self.GANcriterion(scale_preds[-1], True)
        self.loss_G_GAN /= len(fake_preds)
        
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab_chan) * self.lambda_L1
        loss_G = self.loss_G_GAN * self.lambda_GAN + self.loss_G_L1
        return loss_G

    def train(self):
        self.net_G.train()
        self.net_D.train()

    def eval(self):
        self.net_G.eval()
        self.net_D.eval()