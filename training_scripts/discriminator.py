import torch
from torch import nn


class SpectralNorm(nn.Module):
    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class Discriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_layers=5):
        super().__init__()
        layers = [self.get_layers(input_c, num_filters, norm=False)]
        for i in range(1, n_layers):
            nf_prev = num_filters * min(2 ** (i - 1), 8)
            nf = num_filters * min(2**i, 8)
            stride = 1 if i == n_layers - 1 else 2
            layers.append(self.get_layers(nf_prev, nf, s=stride))
        layers.append(self.get_layers(nf, 1, s=1, norm=False, act=False))
        self.model = nn.Sequential(*layers)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        layers = [SpectralNorm(nn.Conv2d(ni, nf, k, s, p, bias=not norm))]
        if norm:
            layers += [nn.InstanceNorm2d(nf)]
        if act:
            layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_layers=5, num_D=3):
        super().__init__()
        self.num_D = num_D
        self.n_layers = n_layers

        for i in range(num_D):
            netD = Discriminator(input_c, num_filters, n_layers)
            setattr(self, f"layer_{i}", netD)

        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False
        )

    def singleD_forward(self, model, input):
        result = [input]
        for i in range(len(model)):
            result.append(model[i](result[-1]))
        return result[1:]

    def forward(self, input):
        result = []
        input_downsampled = input
        for i in range(self.num_D):
            model = getattr(self, f"layer_{i}")
            result.append(self.singleD_forward(model.model, input_downsampled))
            if i != (self.num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result
