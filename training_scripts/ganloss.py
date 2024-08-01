import torch
from torch import nn
import torch.nn.functional as F

class GANLoss(nn.Module):
    def __init__(self, gan_mode="vanilla", real_label=0.9, fake_label=0.1):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError(f"gan_mode {gan_mode} not implemented")

    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_labels(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss