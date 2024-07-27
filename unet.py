import torch
from torch import nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channel, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class UnetBlock(nn.Module):
    def __init__(
        self,
        nf,
        ni,
        submodule=None,
        input_c=None,
        dropout=False,
        innermost=False,
        outermost=False,
        apply_attention=False,
    ):
        super().__init__()
        self.outermost = outermost
        if input_c is None:
            input_c = nf

        downconv = nn.Conv2d(
            input_c, ni, kernel_size=4, stride=2, padding=1, bias=False
        )
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        downnorm = nn.InstanceNorm2d(ni)
        uprelu = nn.ELU(inplace=True)
        upnorm = nn.InstanceNorm2d(nf)

        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                ni, nf, kernel_size=4, stride=2, padding=1, bias=False
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                ni * 2, nf, kernel_size=4, stride=2, padding=1, bias=False
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout:
                up += [nn.Dropout(0.5)]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)
        self.apply_attention = apply_attention
        if apply_attention:
            self.attention = CBAM(nf)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            output = self.model(x)
            if self.apply_attention:
                output = self.attention(output)
            return torch.cat([x, output], 1)


class Unet(nn.Module):
    def __init__(
        self, input_c=1, output_c=2, n_down=8, num_filters=64, init_type="kaiming"
    ):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        for i in range(n_down - 5):
            unet_block = UnetBlock(
                num_filters * 8,
                num_filters * 8,
                submodule=unet_block,
                dropout=False,
                apply_attention=(True),
            )
        out_filters = num_filters * 8
        for i in range(3):
            unet_block = UnetBlock(
                out_filters // 2,
                out_filters,
                submodule=unet_block,
                apply_attention=(True),
            )
            out_filters //= 2
        self.model = UnetBlock(
            output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True
        )
        self.init_weights(init_type)

    def forward(self, x):
        return self.model(x)

    def init_weights(self, init_type="normal", gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                if init_type == "normal":
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError(
                        f"Initialization method {init_type} is not implemented"
                    )
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find("BatchNorm2d") != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
