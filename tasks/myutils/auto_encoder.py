import torch
from torch import nn
from torch.nn import functional as F
import copy
from .types import *

class AutoEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: list = None,
                 img_height=64,
                 img_width=64,
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.img_height = img_height
        self.img_width = img_width

        modules = []
        if hidden_dims is None:
            hidden_dims = [16, 16, 16]
        self.hidden_dims = copy.deepcopy(hidden_dims)
        self.stride = 2
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=self.stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
        self.encoder = nn.Sequential(*modules)
        self.latent = nn.Linear(hidden_dims[-1] * 64, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 64)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=self.stride,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=self.stride,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=1,  # out_channels= 3
                      kernel_size=3, padding=1),
            nn.Tanh())
    def encode(self, input: Tensor) -> Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # # Split the result into mu and var components
        # # of the latent Gaussian distribution
        # mu = self.fc_mu(result)
        # log_var = self.fc_var(result)

        latent = self.latent(result)
        # return [mu, log_var]
        return latent

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        num_feature = self.hidden_dims[-1]
        h_size = int(self.img_height / (self.stride ** len(self.hidden_dims)))
        w_size = int(self.img_width / (self.stride ** len(self.hidden_dims)))
        result = result.view(-1, num_feature, h_size, w_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        # mu, log_var = self.encode(input)
        # z = self.reparameterize(mu, log_var)
        latent = self.encode(input)
        # return  [self.decode(z), input, mu, log_var]
        return [self.decode(latent), input, latent, torch.tensor([0]).cuda()]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        # mu = args[2]
        # log_var = args[3]

        # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)


        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # loss = recons_loss + kld_weight * kld_loss
        # loss = recons_loss
        # return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
        return {'loss': recons_loss, 'Reconstruction_Loss': recons_loss, 'KLD': torch.tensor([0]).cuda()}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]