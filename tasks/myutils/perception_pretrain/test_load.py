from resent_for_perception_pretrain import ResNet18
import torch


if __name__ == "__main__":
    net = ResNet18(num_classes=6).cuda()
    # # 将保存的参数复制到 net3
    # net3.load_state_dict(torch.load('net_params.pkl'))
    # prediction = net3(x)
    net.load_state_dict(torch.load("nn/perception_pretrain_resnet18_classes6.pth"))
    print(net.state_dict())
    net.eval()
