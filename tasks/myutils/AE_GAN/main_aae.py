import random
import trimesh
import torch
import numpy as np
# from models import cvae
from models import aae
from datasets.custom_dataset import Custom_train_dataset, Custom_val_dataset
# from hitdlr_kinematics.hitdlr_layer.hitdlr_layer import HitdlrLayer
# from dataset.DLRdatasetloader import DLRdataset
import torch.backends.cudnn as cudnn
from tqdm import tqdm, trange
from torch.nn import functional as F
import os
import time
from torch.utils.tensorboard import SummaryWriter


seed = 0
set_seed = False
batch_size = 128
distributed = False
workers = 12
epoches = 1
lr = 5e-4
device = 'cuda:3'
adversarial_loss = torch.nn.BCELoss().to(device)
save_seq = 10
load = True
auto_encoder_path = "/home/v-wewei/code/IsaacGym_Preview3/IsaacGymEnvs/isaacgymenvs/tasks/myutils/AE_GAN/nns/aae5/auto_encoder_1639217753.708108.pth"
discriminator_path = "/home/v-wewei/code/IsaacGym_Preview3/IsaacGymEnvs/isaacgymenvs/tasks/myutils/AE_GAN/nns/aae5/discriminator_1639217753.714437.pth"


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
    # loop.set_description(f'Epoch [{epoch}/{epoches}]')
    loop.set_description("Epoch [%d/%d]" % (epoch, epoches))
    for i, batch in enumerate(loop):
        real_images, targets = batch
        real_images = real_images.to(device)
        bs = real_images.size()[0]

        # Adversarial ground truths
        valid_label = torch.full((bs, 1), 1., device=device)
        fake_label = torch.full((bs, 1), 0., device=device)

        # # Configure input
        # real_imgs = grasp_configuration.cuda(non_blocking=True)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = auto_encoder(real_images)[0]

        # recons_loss = F.mse_loss(gen_imgs, real_images)
        # recons_loss = F.smooth_l1_loss(gen_imgs, real_images, beta=1./100)
        recons_loss = F.l1_loss(gen_imgs, real_images)
        # recons_weight = 50.

        # Loss measures generator's ability to fool the discriminator
        adv_loss = adversarial_loss(discriminator(gen_imgs), valid_label)
        g_loss = 0.001 * adv_loss + 0.999 * recons_loss

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_images), valid_label)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake_label)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # print(
        #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        #     % (epoch, epoches, i, len(train_loader), d_loss.item(), g_loss.item())
        # )
        loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item(), recons_loss=recons_loss.item(), adversarial_loss=adv_loss.item())
        batches_done = epoch * len(train_loader) + i
        writer.add_scalar("loss_D/d_loss", d_loss.item(), global_step=batches_done)
        writer.add_scalar("loss_G/g_loss", g_loss.item(), global_step=batches_done)
        writer.add_scalar("loss_G/recons_loss", recons_loss.item(), global_step=batches_done)
        writer.add_scalar("loss_G/adversarial_loss", adv_loss.item(), global_step=batches_done)


# def valiate(val_loader):
#     generator.eval()
#     discriminator.eval()
#     with torch.no_grad():
#
#         for i, batch in enumerate(tqdm(val_loader)):
#             z = torch.randn(1, 16).to(device)
#             output = generator(z).squeeze().detach().cpu().numpy()
#
#             object_vertices, grasp_configuration, pose = batch
#             object_vertices = object_vertices.cuda(non_blocking=True)
#
#             object_vertices_array = object_vertices.squeeze().detach().cpu().numpy().transpose(1, 0)
#             pos = output[:3] / 100
#             quat = output[3:7] / np.linalg.norm(output[3:7])
#             tranlation_matrix = trimesh.transformations.translation_matrix(pos)
#             rotation_matrix = trimesh.transformations.quaternion_matrix(quat)
#             T = trimesh.transformations.concatenate_matrices(tranlation_matrix, rotation_matrix)
#             pose = torch.from_numpy(T).to(device).reshape(-1, 4, 4).float()
#             theta_array = np.deg2rad(output[7:]).astype(np.float32)
#             theta = torch.from_numpy(theta_array).to(device).reshape(-1, 20)
#             # print(pose, theta)
#             # exit()
#             meshes = hit.get_forward_hand_mesh(pose, theta, save_mesh=False, path='./output_mesh')
#             pc = trimesh.PointCloud(object_vertices_array, colors=[0, 0, 1])
#             scene = trimesh.Scene([pc, meshes[0]])
#             scene.show()


if __name__ == '__main__':
    if set_seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True

    auto_encoder = aae.AutoEncoder(in_channels=1, latent_dim=100, hidden_dims=[32, 32, 32]).to(device)
    if load:
        auto_encoder.load_state_dict(torch.load(auto_encoder_path, map_location=torch.device(device)))
    discriminator = aae.Discriminator().to(device)
    if load:
        discriminator.load_state_dict(torch.load(discriminator_path, map_location=torch.device(device)))
    train_dataset = Custom_train_dataset(root="../image_tensors/",
                                         split="train",
                                         transform=None,
                                         device='cpu',
                                         download=False)
    val_dataset = Custom_val_dataset(root="../image_tensors/",
                                     split="train",
                                     transform=None,
                                     device='cpu',
                                     download=False)
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
        val_dataset, batch_size=1, shuffle=False,
        sampler=None, persistent_workers=True, num_workers=workers)
    # valiate(val_loader)
    if not os.path.exists("summaries"):
        os.mkdir("summaries")
    if not os.path.exists("summaries/aae"):
        os.mkdir("summaries/aae")
    writer = SummaryWriter('summaries/aae')
    for epoch in range(epoches):
        adjust_learning_rate(optimizer_D, epoch, lr)
        adjust_learning_rate(optimizer_G, epoch, lr)
        train_epoch(train_loader, epoch, epoches)
        if epoch % save_seq == 0:
            if not os.path.exists("nns"):
                os.mkdir("nns")
            if not os.path.exists("nns/aae"):
                os.mkdir("nns/aae")
            torch.save(auto_encoder.state_dict(), "nns/aae/auto_encoder.pth")
            torch.save(discriminator.state_dict(), "nns/aae/discriminator.pth")
    # valiate(val_loader)

    if not os.path.exists("nns"):
        os.mkdir("nns")
    if not os.path.exists("nns/aae"):
        os.mkdir("nns/aae")
    torch.save(auto_encoder.state_dict(), "nns/aae/auto_encoder_%f.pth" % time.time())
    torch.save(discriminator.state_dict(), "nns/aae/discriminator_%f.pth" % time.time())



