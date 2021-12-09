import torch
from models import ae_gan
import yaml
from PIL import Image
import numpy as np


if __name__ == "__main__":
    model = ae_gan.AutoEncoder(in_channels=1, latent_dim=100, hidden_dims=[32, 32, 32]).cuda()
    model.load_state_dict(torch.load("nns/ae_gan/auto_encoder.pth"))
    image_tensors = torch.load("/home/wsz/IsaacGym_Preview_3_Package/IsaacGymEnvs/isaacgymenvs/tasks/myutils/image_tensors/image_tensors.pt")[-2:]
    recon = model.generate(image_tensors)
    imgs = []


    for image_tenosr in image_tensors:
        img = image_tenosr.detach().permute(1, 2, 0).squeeze(-1).cpu().numpy()
        img = (img + 1.0) / 2.0
        img *= 255.0
        img = Image.fromarray(img.astype(np.uint8), mode="L")
        imgs.append(img)
        img.show()
    for image_tenosr in recon:
        img = image_tenosr.detach().permute(1, 2, 0).squeeze(-1).cpu().numpy()
        img = (img + 1.0) / 2.0
        img *= 255.0
        img = Image.fromarray(img.astype(np.uint8), mode="L")
        imgs.append(img)
        img.show()

