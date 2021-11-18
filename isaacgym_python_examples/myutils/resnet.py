import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models
from PIL import Image
class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        #self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth'))
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        # x = x.view(x.size(0), x.size(1))
        return x


if __name__ == "__main__":
    # model = models.resnet34(pretrained=True)
    # list(model.modules())   # to inspect the modules of your model
    # my_model = nn.Sequential(*list(model.modules())[:-1])   # strips off last linear layer
    # print(my_model)
    model = models.resnet34(pretrained=True)
    newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
    # print(newmodel)
    # img_dir0 = "/home/v-wewei/code/isaacgym/python/examples/yumi_image/rgb_env0.png"
    # img_dir1 = "/home/v-wewei/code/isaacgym/python/examples/yumi_image/rgb_env1.png"
    transform = transforms.Compose([            #[1]
     transforms.Resize(472),                    #[2]
     transforms.CenterCrop(472),                #[3]
     transforms.ToTensor(),                     #[4]
     transforms.Normalize(                      #[5]
     mean=[0.485, 0.456, 0.406],                #[6]
     std=[0.229, 0.224, 0.225]                  #[7]
     )])
    # imgs_t = []
    # img0 = Image.open(img_dir0).convert('RGB')
    # # imgs.append(img0)
    # img1 = Image.open(img_dir1).convert('RGB')
    # # imgs.append(img1)
    # img_t0 = transform(img0)
    # batch_t0 = torch.unsqueeze(img_t0, 0)
    # imgs_t.append(batch_t0)
    # img_t1 = transform(img1)
    # batch_t1 = torch.unsqueeze(img_t1, 0)
    # imgs_t.append(batch_t1)

    # =========================
    num_envs = 16
    imgs = []
    for i in range(num_envs):
        img_dir = "/home/v-wewei/code/isaacgym/python/examples/yumi_image/rgb_env%d.png" % i
        img = Image.open(img_dir)
        print(img)
        img = Image.open(img_dir).convert('RGB')
        img_trans = transform(img)
        batch_img = torch.unsqueeze(img_trans, 0)
        imgs.append(batch_img)
    imgs_tensor = torch.cat(imgs)

    # =========================
    print(imgs_tensor.size())

    out = newmodel(imgs_tensor)
    # print(out)
    print(out.size())


