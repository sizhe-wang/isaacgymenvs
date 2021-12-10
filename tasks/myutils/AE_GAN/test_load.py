import torch
from models import ae_gan
import yaml
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# import pylab


if __name__ == "__main__":
    model_path = "/home/wsz/IsaacGym_Preview_3_Package/IsaacGymEnvs/isaacgymenvs/tasks/myutils/AE_GAN/nns/ae_gan/ae_gan/auto_encoder_1639070062.656193.pth"
    model = ae_gan.AutoEncoder(in_channels=1, latent_dim=100, hidden_dims=[32, 32, 32]).cuda()
    model.load_state_dict(torch.load(model_path))
    image_tensors = torch.load("../image_tensors/image_tensors.pt")[-8:]
    recon = model.generate(image_tensors)
    imgs = []

    # count = 1
    # # plt.subplots_adjust(left=0., bottom=0., top=0.001, right=0.001, hspace=0.001, wspace=0.001)
    # for image_tenosr in image_tensors:
    #     img = image_tenosr.detach().permute(1, 2, 0).squeeze(-1).cpu().numpy()
    #     img = (img + 1.0) / 2.0
    #     img *= 255.0
    #     # img = Image.fromarray(img.astype(np.uint8), mode="L")
    #     # imgs.append(img)
    #     # img.show()
    #
    #     plt.subplot(2, 8, count)
    #     plt.imshow(img, cmap='Greys_r')
    #     plt.axis('off')
    #     count += 1
    #
    # for image_tenosr in recon:
    #     img = image_tenosr.detach().permute(1, 2, 0).squeeze(-1).cpu().numpy()
    #     img = (img + 1.0) / 2.0
    #     img *= 255.0
    #     # img = Image.fromarray(img.astype(np.uint8), mode="L")
    #     # imgs.append(img)
    #     # img.show()
    #     plt.subplot(2, 8, count)
    #     plt.imshow(img, cmap='Greys_r')
    #     plt.axis('off')
    #     count += 1
    # plt.tight_layout()
    # plt.show()

    count = 0
    fig, axes = plt.subplots(2, 8, sharex='all', sharey='all', figsize=(14, 4))
    for image_tenosr in image_tensors:
        img = image_tenosr.detach().permute(1, 2, 0).squeeze(-1).cpu().numpy()
        img = (img + 1.0) / 2.0
        img *= 255.0

        axes[0, count].imshow(img, cmap='Greys_r')
        axes[0, count].axis('off')
        count += 1

    count = 0
    for image_tenosr in recon:
        img = image_tenosr.detach().permute(1, 2, 0).squeeze(-1).cpu().numpy()
        img = (img + 1.0) / 2.0
        img *= 255.0

        axes[1, count].imshow(img, cmap='Greys_r')
        axes[1, count].axis('off')
        count += 1

    # 调节子图直接的距离
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
