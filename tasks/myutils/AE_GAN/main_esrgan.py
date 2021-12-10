import random
import trimesh
import torch
import numpy as np
# from models import cvae
from models import ae_gan
from models.RRDBNet_arch import GeneratorRRDB, Discriminator, FeatureExtractor
from datasets.custom_dataset import Custom_train_dataset, Custom_val_dataset
# from hitdlr_kinematics.hitdlr_layer.hitdlr_layer import HitdlrLayer
# from dataset.DLRdatasetloader import DLRdataset
import torch.backends.cudnn as cudnn
from tqdm import tqdm, trange
from torch.nn import functional as F
import os
import time


seed = 0
set_seed = False
batch_size = 128
distributed = False
workers = 12
epoches = 40
lr = 5e-4
device = 'cuda:0'
save_seq = 10
load = False
auto_encoder_path = "nns/esrgan/auto_encoder.pth"
discriminator_path = "nns/esrgan/discriminator.pth"
in_channel = 1
hr_shape = (64, 64)
warmup_batches = 1      # "number of batches with pixel-wise loss only"
lambda_adv = 5e-3   # "adversarial loss weight"
lambda_pixel = 1e-2     # "pixel-wise loss weight"

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)
adversarial_loss = torch.nn.BCELoss().to(device)


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_epoch(train_loader, epoch, epoches):
    # model.train()
    auto_encoder.train()
    discriminator.train()
    loop = tqdm(train_loader, total=len(train_loader), leave=True)
    loop.set_description("Epoch [%d/%d]" % (epoch, epoches))
    for i, batch in enumerate(loop):
        batches_done = epoch * len(train_loader) + i
        real_images, targets = batch
        real_images = real_images.to(device)
        bs = real_images.size()[0]

        # Adversarial ground truths
        valid_label = torch.full((bs, *discriminator.output_shape), 1., device=device)
        fake_label = torch.full((bs, *discriminator.output_shape), 0., device=device)

        # # Configure input
        # real_imgs = grasp_configuration.cuda(non_blocking=True)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = auto_encoder(real_images)[0]

        # # recons_loss = F.mse_loss(gen_imgs, real_images)
        # recons_loss = F.smooth_l1_loss(gen_imgs, real_images, beta=1./100)
        # recons_weight = 50.
        # Measure pixel-wise loss against ground truth
        # loss_pixel = criterion_pixel(gen_imgs, real_images)
        loss_pixel = F.smooth_l1_loss(gen_imgs, real_images, beta=1./100)
        if batches_done < warmup_batches:
            # Warm-up (pixel-wise loss only)
            loss_pixel.backward()
            optimizer_G.step()
            loop.set_postfix(loss_pixel=loss_pixel.item())
            continue

        # Extract validity predictions from discriminator
        pred_real = discriminator(real_images).detach()
        pred_fake = discriminator(gen_imgs)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid_label)

        # Total generator loss
        loss_G = lambda_adv * loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        pred_real = discriminator(real_images)
        pred_fake = discriminator(gen_imgs.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid_label)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake_label)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # print(
        #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        #     % (epoch, epoches, i, len(train_loader), d_loss.item(), g_loss.item())
        # )
        loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item(),
                         loss_GAN=loss_GAN.item(), loss_pixel=loss_pixel.item())
        # batches_done = epoch * len(train_loader) + i

if __name__ == '__main__':
    if set_seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True

    auto_encoder = ae_gan.AutoEncoder(in_channels=1, latent_dim=100, hidden_dims=[32, 32, 32]).to(device)
    if load:
        auto_encoder.load_state_dict(torch.load(auto_encoder_path))
    discriminator = Discriminator(input_shape=(in_channel, *hr_shape)).to(device)
    if load:
        discriminator.load_state_dict(torch.load(discriminator_path))

    train_dataset = Custom_train_dataset(root="../image_tensors/",
                                         split="train",
                                         transform=None,
                                         download=False,
                                         device='cpu',
                                         percentage=0.1)
    val_dataset = Custom_val_dataset(root="../image_tensors/",
                                     split="train",
                                     transform=None,
                                     download=False,
                                     device='cpu',
                                     percentage=0.1)
    optimizer_G = torch.optim.Adam(auto_encoder.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), drop_last=False,
        sampler=train_sampler, persistent_workers=True, num_workers=workers)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, sampler=None, persistent_workers=True, num_workers=workers)
    # valiate(val_loader)

    for epoch in range(epoches):
        adjust_learning_rate(optimizer_D, epoch, lr)
        adjust_learning_rate(optimizer_G, epoch, lr)
        train_epoch(train_loader, epoch, epoches)
        if epoch % save_seq == 0:
            if not os.path.exists("nns/esrgan"):
                os.mkdir("nns/esrgan")
            torch.save(auto_encoder.state_dict(), "nns/esrgan/auto_encoder.pth")
            torch.save(discriminator.state_dict(), "nns/esrgan/discriminator.pth")
    # valiate(val_loader)

    if not os.path.exists("nns"):
        os.mkdir("nns")
    if not os.path.exists("nns/esrgan"):
        os.mkdir("nns/esrgan")
    torch.save(auto_encoder.state_dict(), "nns/esrgan/auto_encoder_%f.pth" % time.time())
    torch.save(discriminator.state_dict(), "nns/esrgan/discriminator_%f.pth" % time.time())



