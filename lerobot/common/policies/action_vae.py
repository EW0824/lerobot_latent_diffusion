# lerobot/common/policies/latent_vae.py
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Optional

class ActionVAE(nn.Module):

    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 input_dim: int = 4,
                 latent_dim: int = 2,
                 hidden_dims: Optional[List[int]] = None,
                 beta: float = 4.0,
                 gamma: float = 1000.0,
                 max_capacity: float = 25.0,
                 Capacity_max_iter: int = int(1e5),
                 loss_type: str = 'H',
                 use_skip_connections: bool = True,  # Add skip connections for better gradients
                 dropout_rate: float = 0.1,  # Add dropout for regularization
                 **kwargs) -> None:
        super(ActionVAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.use_skip_connections = use_skip_connections

        # Set default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]  # Better default for robotics

        # Build Encoder with skip connections
        self.encoder_layers = nn.ModuleList()
        current_dim = input_dim
        
        for i, h_dim in enumerate(hidden_dims):
            layer = nn.Sequential(
                nn.Linear(current_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            )
            self.encoder_layers.append(layer)
            current_dim = h_dim
        
        # Latent space projections
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder with skip connections
        self.decoder_layers = nn.ModuleList()
        current_dim = latent_dim
        
        # Reverse the hidden dimensions for decoder
        decoder_hidden_dims = hidden_dims[::-1]
        
        for i, h_dim in enumerate(decoder_hidden_dims):
            layer = nn.Sequential(
                nn.Linear(current_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            )
            self.decoder_layers.append(layer)
            current_dim = h_dim
        
        # Final output layer with careful initialization
        self.final_layer = nn.Sequential(
            nn.Linear(current_dim, input_dim),
            nn.Tanh()  # Assuming normalized input data in [0, 1]
        )
        
        # Initialize weights for stable training
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x input_dim]
        :return: (List[Tensor]) List of latent codes [mu, log_var]
        """
        x = input
        encoder_outputs = []
        
        # Forward through encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
            encoder_outputs.append(x)

        # Split the result into mu and var components
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes onto the data space.
        :param z: (Tensor) [B x latent_dim]
        :return: (Tensor) [B x input_dim]
        """
        x = z
        
        # Forward through decoder layers
        for layer in self.decoder_layers:
            x = layer(x)
        
        result = self.final_layer(x)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x latent_dim]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x latent_dim]
        :return: (Tensor) [B x latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        """
        Forward pass through the VAE.
        :param input: (Tensor) [B x input_dim]
        :return: (List[Tensor]) [reconstruction, input, mu, log_var]
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return [reconstruction, input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples: int,
               current_device: torch.device, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        data space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Device) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input vector x, returns the reconstructed vector
        :param x: (Tensor) [B x input_dim]
        :return: (Tensor) [B x input_dim]
        """
        return self.forward(x)[0]

    def encode_to_latent(self, input: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space (deterministic, using mean).
        Useful for diffusion policy training.
        """
        mu, log_var = self.encode(input)
        return mu  # Use mean for deterministic encoding

    def encode_with_variance(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space returning both mean and variance.
        Useful for uncertainty estimation in diffusion models.
        :param x: (Tensor) [B x input_dim]
        :return: (Tuple[Tensor, Tensor]) [mu, log_var]
        """
        mu, log_var = self.encode(x)
        return mu, log_var

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """
        Interpolate between two input vectors in latent space
        :param x1: (Tensor) [1 x input_dim] First vector
        :param x2: (Tensor) [1 x input_dim] Second vector  
        :param num_steps: (int) Number of interpolation steps
        :return: (Tensor) [num_steps x input_dim] Interpolated vectors
        """
        mu1, _ = self.encode(x1)
        mu2, _ = self.encode(x2)
        
        # Create interpolation coefficients
        alphas = torch.linspace(0, 1, num_steps).to(x1.device).view(-1, 1)
        
        # Interpolate in latent space
        z_interp = (1 - alphas) * mu1 + alphas * mu2
        
        # Decode back to data space
        return self.decode(z_interp)

    def get_latent_stats(self, dataloader: torch.utils.data.DataLoader, device: torch.device) -> dict:
        """
        Compute latent space statistics for the entire dataset.
        Useful for conditioning diffusion models.
        """
        self.eval()
        all_mus = []
        all_log_vars = []
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(device)
                mu, log_var = self.encode(x)
                all_mus.append(mu)
                all_log_vars.append(log_var)
        
        all_mus = torch.cat(all_mus, dim=0)
        all_log_vars = torch.cat(all_log_vars, dim=0)
        
        return {
            'latent_mean': all_mus.mean(dim=0),
            'latent_std': all_mus.std(dim=0),
            'latent_min': all_mus.min(dim=0)[0],
            'latent_max': all_mus.max(dim=0)[0],
            'avg_log_var': all_log_vars.mean(dim=0)
        }

    def set_beta(self, beta: float) -> None:
        """Set the beta value for KL divergence weighting."""
        self.beta = beta
    
    def get_current_beta(self) -> float:
        """Get the current beta value."""
        return self.beta

    def sample_latent(self, input: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Sample from the latent distribution.
        Useful for generating multiple latent codes from same input.
        """
        mu, log_var = self.encode(input)
        if num_samples == 1:
            return self.reparameterize(mu, log_var)
        else:
            # Generate multiple samples
            batch_size = input.shape[0]
            samples = []
            for _ in range(num_samples):
                sample = self.reparameterize(mu, log_var)
                samples.append(sample)
            return torch.stack(samples, dim=1)  # [batch, num_samples, latent_dim]

    def get_latent_statistics(self, input: torch.Tensor) -> dict:
        """Get statistics of the latent distribution."""
        mu, log_var = self.encode(input)
        std = torch.exp(0.5 * log_var)
        
        return {
            'mean': mu.mean(dim=0).detach(),
            'std': std.mean(dim=0).detach(),
            'total_var': torch.sum(std, dim=1).mean().detach(),
            'kl_per_dim': -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).mean(dim=0).detach()
        }

    def freeze_encoder(self):
        """Freeze encoder parameters for decoder-only fine-tuning."""
        for layer in self.encoder_layers:
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.fc_mu.parameters():
            param.requires_grad = False
        for param in self.fc_var.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for layer in self.encoder_layers:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.fc_mu.parameters():
            param.requires_grad = True
        for param in self.fc_var.parameters():
            param.requires_grad = True 