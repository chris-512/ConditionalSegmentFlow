import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, model_urls
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class ResNet50FPN(nn.Module):
    def __init__(self, input_width=224, input_height=224):
        super(ResNet50FPN, self).__init__()

        self.backbone = resnet_fpn_backbone(
            'resnet50', pretrained=True, trainable_layers=3)
        # num_features = self.net.fc.in_features
        # self.net.fc = nn.Linear(num_features, 1024)
        # self.net = self.net.cuda()

    def forward(self, x):
        return self.backbone(x)


if __name__ == '__main__':
    net = ResNet50FPN(input_width=256, input_height=256)
    x = torch.rand(1, 3, 256, 256)

    outputs = net(x)

    net_outputs = []
    for key, output in outputs.items():
        net_outputs.append(nn.AvgPool2d(kernel_size=output.size(2))(
            output))
    net_outputs = torch.cat(net_outputs, dim=0)
    net_out = net_outputs.view(1, -1)
    """
    net = resnet50(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 1024)
    net = net.cuda()

    x = torch.randn(1, 3, 224, 224).cuda()
    y = net(x)

    """
