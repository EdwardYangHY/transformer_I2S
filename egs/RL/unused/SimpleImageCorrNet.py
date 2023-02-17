import torch.nn as nn
import torchvision.models as models
import torch
import yaml

with open('../../config.yml', 'r') as yml:
    config = yaml.safe_load(yml)

class SimpleImageCorrNet(nn.Module):
    def __init__(self, num_class = 50, pre_trained=False):
        super(SimpleImageCorrNet, self).__init__()
        self.image_net = models.resnet50(num_classes=num_class, pretrained=pre_trained)
        self.image_net.fc = nn.Identity()
        # self.image_net.load_state_dict(torch.load("dino_resnet50_pretrain.pth"))
        self.image_net.load_state_dict(torch.load("../../model/dino_resnet50_pretrain.pth"))
        self.image_net.fc = nn.Linear(in_features=2048, out_features=num_class, bias=True)


    def _visual_extract(self, v):
        return self.image_net(v)


    def forward(self, image_feat):
        image_feat = self._visual_extract(image_feat)
        return image_feat