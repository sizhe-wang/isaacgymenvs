'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet(nn.Module):
    def __init__(self, layers=18, expansion=1, num_classes=10, learning_rate=1e-3):
        super(ResNet, self).__init__()
        # self.model = torch.nn.Sequential(*(list(models.resnet18(pretrained=False, num_classes=100).children())[:-1]))
        self._network = eval('models.resnet{}'.format(layers))(pretrained=False, num_classes=num_classes)
        self.lr = learning_rate


    def create_scheduler(self, milestones=None, gamma=None):
        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, milestones=milestones, gamma=gamma)

    def create_optimzer(self):
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=self.lr)

    def compute_loss(self, output, target):
        # loss calculation
        # print("in loss, output", output[0])
        # print("in loss, target", target[0])
        loss = F.l1_loss(output, target)
        return loss

    def train_network(self, input_data=None, target=None):
        output = self.forward(input_data)
        # loss computation
        loss = self.compute_loss(output, target)
        print(loss)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # self._scheduler.step()
        return output

    def inference_network(self, input_data=None, target=None):
        output = self.forward(input_data)
        # loss computation
        return output

    def forward(self, x):
        out = self._network(x)
        # print("in forward, out", out[0])
        # print("in forward, x", x[0])
        return out


def ResNet18(**kwargs):
    return ResNet(layers=18, expansion=1, **kwargs)


def ResNet34(**kwargs):
    return ResNet(layers=34, expansion=1, **kwargs)


def ResNet50(**kwargs):
    return ResNet(layers=50, expansion=4, **kwargs)


def ResNet101(**kwargs):
    return ResNet(layers=101, expansion=4, **kwargs)


def ResNet152(**kwargs):
    return ResNet(layers=152, expansion=4, **kwargs)


def test():
    net = ResNet18(num_classes=4).cuda()

    net.create_optimzer()
    net.create_scheduler()
    net.train()
    for i in range(100):
        input_data = torch.randn(16, 3, 256, 256).cuda()
        target = torch.randn(16, 4).cuda()
        y = net.train_network(input_data, target).detach()


if __name__ == "__main__":
    test()
