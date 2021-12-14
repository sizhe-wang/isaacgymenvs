import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from isaacgymenvs.tasks.myutils.AE_GAN.models.aae import AutoEncoder, Discriminator
import copy
import os
from tqdm import tqdm
import time


class AAE(torch.nn.Module):
    def __init__(self, in_channels=1, latent_dim=100, hidden_dims=[32, 32, 32], device='cuda:0'):
        super().__init__()
        self.device = device
        self.auto_encoder = AutoEncoder(in_channels=in_channels,
                                        latent_dim=latent_dim,
                                        hidden_dims=hidden_dims).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.adversarial_loss = torch.nn.BCELoss().to(self.device)
        if not os.path.exists("summaries"):
            os.mkdir("summaries")
        if not os.path.exists("summaries/aae"):
            os.mkdir("summaries/aae")
        if not os.path.exists("nns"):
            os.mkdir("nns")
        if not os.path.exists("nns/aae"):
            os.mkdir("nns/aae")
        self.writer = SummaryWriter('summaries/aae')

    def create_optimizer(self, lr=5e-4, betas=(0.5, 0.999)):
        self.optimizer_G = torch.optim.Adam(self.auto_encoder.parameters(), lr=lr, betas=betas)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)

    def create_scheduler(self, gamma=1.):
        self.scheduler_G = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_G, gamma=gamma)
        self.scheduler_D = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_D, gamma=gamma)

    def train_epoch(self, real_images, n_batch, save_seq=10, warmup=-1):
        # model.train()
        self.auto_encoder.train()
        self.discriminator.train()

        real_images = real_images.to(self.device)
        bs = real_images.size()[0]

        # Adversarial ground truths
        valid_label = torch.full((bs, 1), 1., device=self.device)
        fake_label = torch.full((bs, 1), 0., device=self.device)

        # -----------------
        #  Train Generator
        # -----------------

        self.optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = self.auto_encoder(real_images)[0]

        # recons_loss = F.mse_loss(gen_imgs, real_images)
        # recons_loss = F.smooth_l1_loss(gen_imgs, real_images, beta=1./100)
        recons_loss = F.l1_loss(gen_imgs, real_images)
        # recons_weight = 50.

        # Loss measures generator's ability to fool the discriminator

        adv_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid_label)
        g_loss = 0.001 * adv_loss + 0.999 * recons_loss

        g_loss.backward()
        self.optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        self.optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = self.adversarial_loss(self.discriminator(real_images), valid_label)
        fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake_label)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        self.optimizer_D.step()

        if warmup >= 0 and n_batch > warmup:
            self.scheduler_G.step()
            self.scheduler_D.step()
            print("step scheduler", n_batch)
        # print(
        #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        #     % (epoch, epoches, i, len(train_loader), d_loss.item(), g_loss.item())
        # )

        batches_done = n_batch
        self.writer.add_scalar("loss_D/d_loss", d_loss.item(), global_step=batches_done)
        self.writer.add_scalar("loss_G/g_loss", g_loss.item(), global_step=batches_done)
        self.writer.add_scalar("loss_G/recons_loss", recons_loss.item(), global_step=batches_done)
        self.writer.add_scalar("loss_G/adversarial_loss", adv_loss.item(), global_step=batches_done)
        # save state dict
        if n_batch % save_seq == 0:
            torch.save(self.auto_encoder.state_dict(), "nns/aae/auto_encoder_%d.pth" % n_batch)
            torch.save(self.discriminator.state_dict(), "nns/aae/discriminator_%d.pth" % n_batch)


if __name__ == '__main__':
    aae_model = AAE(device='cuda:1')
    image_tensors = torch.load('image_tensors/image_tensors.pt', map_location=torch.device(aae_model.device))
    aae_model.auto_encoder.load_state_dict(torch.load('/home/v-wewei/code/IsaacGym_Preview3/IsaacGymEnvs/isaacgymenvs/tasks/myutils/AE_GAN/nns/aae5/auto_encoder_1639217753.708108.pth', map_location=torch.device(aae_model.device)))
    aae_model.discriminator.load_state_dict(torch.load('/home/v-wewei/code/IsaacGym_Preview3/IsaacGymEnvs/isaacgymenvs/tasks/myutils/AE_GAN/nns/aae5/discriminator_1639217753.714437.pth', map_location=torch.device(aae_model.device)))
    aae_model.create_optimizer()
    aae_model.create_scheduler()
    num_envs = 1024
    for step_counter in range(20):
        input_data = image_tensors[step_counter*num_envs:(step_counter+1)*num_envs]
        aae_model.train_epoch(input_data, step_counter, 5, warmup=-1)
        print("run %d" % step_counter)

    torch.save(aae_model.auto_encoder.state_dict(), "nns/aae/auto_encoder_%d.pth" % time.time())
    torch.save(aae_model.discriminator.state_dict(), "nns/aae/discriminator_%d.pth" % time.time())