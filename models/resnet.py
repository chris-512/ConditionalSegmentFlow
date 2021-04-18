import torch
import torch.nn as nn 
from torchvision.models.resnet import resnet50, model_urls

if __name__ == '__main__':
    net = resnet50(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 1024)
    net = net.cuda()

    x = torch.randn(1, 3, 224, 224).cuda()
    y = net(x)
    import pdb; pdb.set_trace()