import torch
from torch import nn
import torch.nn.functional as F


class CrissCrossAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.channel_in = in_dim
        self.channel_out = in_dim // 8
        # Combined QKV projection
        self.qkv_conv = nn.Conv2d(in_dim, 3 * self.channel_out, 1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.out_conv = nn.Conv2d(in_dim // 8, in_dim, 1)

    def forward(self, x):
        B, C, H, W = x.size()
        
        # Combined QKV projection
        qkv = self.qkv_conv(x)
        query, key, value = qkv.chunk(3, dim=1)
        
        # Horizontal attention
        query_h = query.permute(0, 3, 1, 2).contiguous().view(B*W, self.channel_out, H)
        key_h = key.permute(0, 3, 1, 2).contiguous().view(B*W, self.channel_out, H)
        value_h = value.permute(0, 3, 1, 2).contiguous().view(B*W, self.channel_out, H)
        
        energy_h = torch.bmm(query_h, key_h.transpose(1, 2))
        attn_h = F.softmax(energy_h, dim=-1)
        out_h = torch.bmm(attn_h, value_h)
        
        # Vertical attention
        query_v = query.permute(0, 2, 1, 3).contiguous().view(B*H, self.channel_out, W)
        key_v = key.permute(0, 2, 1, 3).contiguous().view(B*H, self.channel_out, W)
        value_v = value.permute(0, 2, 1, 3).contiguous().view(B*H, self.channel_out, W)
        
        energy_v = torch.bmm(query_v, key_v.transpose(1, 2))
        attn_v = F.softmax(energy_v, dim=-1)
        out_v = torch.bmm(attn_v, value_v)
        
        # Reshape and combine
        out_h = out_h.view(B, W, self.channel_out, H).permute(0, 2, 3, 1)
        out_v = out_v.view(B, H, self.channel_out, W).permute(0, 2, 1, 3)
        
        out = out_h + out_v
        out = self.gamma * out
        
        # Project back to the original channel dimension
        out = self.out_conv(out)
        
        return out + x


class UnetBlock(nn.Module):
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        if input_c is None: input_c = nf
        downconv = nn.Conv2d(input_c, ni, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)
        
        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout: up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        
        self.model = nn.Sequential(*model)
        self.use_attention = not outermost and nf <= 512
        if self.use_attention:
            self.attention = CrissCrossAttention(nf)
        
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            output = self.model(x)
            if self.use_attention:
                output = self.attention(output)
            return torch.cat([x, output], 1)


class Unet(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)

        for i in range(n_down - 5):
            unet_block = UnetBlock(
                num_filters * 8, num_filters * 8, submodule=unet_block, dropout=False
            )

        out_filters = num_filters * 8
        for i in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2

        self.model = UnetBlock(
            output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True
        )

    def forward(self, x):
        return self.model(x)
