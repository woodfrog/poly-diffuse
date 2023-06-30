import torch
import torch.nn as nn
from torchvision import models


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

        self.strides = [8, 16, 32]
        self.num_channels = [512, 1024, 2048]

    def forward(self, inputs):
        layer0 = self.layer0(inputs)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        xs = {"0": layer2, "1": layer3, "2": layer4}
        all_feats = {'layer0': layer0, 'layer1': layer1, 'layer2': layer2,
                     'layer3': layer3, 'layer4': layer4}

        mask = torch.zeros(inputs.shape)[:, 0, :, :].to(layer4.device)
        return xs, mask, all_feats
