"""Faraday baseline model: VAE + Gaussian Mixture Model composition.

This implementation follows the Faraday philosophy of replacing the standard Gaussian 
prior in VAE latent space with a Gaussian Mixture Model. The model works by:
1. Loading a pre-trained VAE checkpoint
2. Fitting a GMM in the VAE's latent space
3. Using the GMM for improved sampling

The model maintains the same API as VAEModelPL for seamless integration with the 
existing training and evaluation pipeline.
"""

from collections.abc import Callable

import pytorch_lightning as pl
import torch
from einops import rearrange
from torch import Tensor, nn

from ....utils.configuration import TrainConfig
from .vae import VAEModelPL


NUM_GMM_COMPONENTS = 10  # Hardcoded number of GMM components


class GaussianMixtureModel(nn.Module):
    """PyTorch-friendly Gaussian Mixture Model with batch-wise EM training."""
    
    weights: torch.Tensor
    means: torch.Tensor
    precision_cholesky: torch.Tensor
    covariances: torch.Tensor
    nll: torch.Tensor
    
    def __init__(
        self,
        num_components: int,
        num_features: int,
        reg_covar: float = 1e-6,
    ):
        super().__init__()
        self.num_components = num_components
        self.num_features = num_features
        self.reg_covar = reg_covar
        
        # Initialize model params as buffers
        weights_shape = torch.Size([self.num_components])
        means_shape = torch.Size([self.num_components, self.num_features])
        precision_cholesky_shape = torch.Size(
            [self.num_components, self.num_features, self.num_features]
        )
        covariances_shape = torch.Size(
            [self.num_components, self.num_features, self.num_features]
        )
        nll_shape = torch.Size([1])
        
        self.register_buffer("weights", torch.empty(weights_shape))
        self.register_buffer("means", torch.empty(means_shape))
        self.register_buffer("precision_cholesky", torch.empty(precision_cholesky_shape))
        self.register_buffer("covariances", torch.empty(covariances_shape))
        self.register_buffer("nll", torch.empty(nll_shape))
        self.initialised = False
        
    def initialise(self, X: Tensor):
        """Initialize GMM parameters using K-means++ style initialization."""
        n_samples = X.shape[0]
        
        # Random initialization for weights (uniform)
        self.weights.fill_(1.0 / self.num_components)
        
        # K-means++ style initialization for means
        indices = torch.randperm(n_samples)[:self.num_components]
        self.means.copy_(X[indices])
        
        # Initialize covariances as identity matrices scaled by data variance
        data_var = torch.var(X, dim=0).mean()
        identity = torch.eye(self.num_features, device=X.device)
        covariances = identity.unsqueeze(0).repeat(self.num_components, 1, 1) * data_var
        self.covariances.copy_(covariances)
        
        # Compute precision matrices
        precision_matrices = torch.linalg.inv(self.covariances + torch.eye(self.num_features, device=X.device) * self.reg_covar)
        self.precision_cholesky.copy_(torch.linalg.cholesky(precision_matrices))
        
        self.initialised = True
        
    def _estimate_log_gaussian_prob(self, X: Tensor) -> Tensor:
        """Estimate log Gaussian probability for each component."""
        n_samples, n_features = X.shape
        log_prob = torch.zeros(n_samples, self.num_components, device=X.device)
        
        for k in range(self.num_components):
            # Use torch.distributions for numerical stability
            try:
                dist = torch.distributions.MultivariateNormal(
                    self.means[k], self.covariances[k]
                )
                log_prob[:, k] = dist.log_prob(X)
            except RuntimeError:
                # Fallback for numerical issues
                log_prob[:, k] = -1e6  # Very low probability
                
        return log_prob
        
    def e_step(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """E-step: compute responsibilities."""
        log_prob = self._estimate_log_gaussian_prob(X)  # [n_samples, n_components]
        log_weights = torch.log(self.weights + 1e-8)  # [n_components]
        
        # Weighted log probabilities
        weighted_log_prob = log_prob + log_weights.unsqueeze(0)  # [n_samples, n_components]
        
        # Log-sum-exp for numerical stability
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1)  # [n_samples]
        log_resp = weighted_log_prob - log_prob_norm.unsqueeze(1)  # [n_samples, n_components]
        
        return torch.mean(log_prob_norm), log_resp
        
    def m_step(self, X: Tensor, log_responsibilities: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """M-step: update parameters based on responsibilities."""
        resp = torch.exp(log_responsibilities)  # [n_samples, n_components]
        
        # Update weights
        nk = torch.sum(resp, dim=0) + 1e-6  # Add small epsilon for stability
        weights = nk / torch.sum(nk)
        
        # Update means
        means = torch.sum(resp.unsqueeze(2) * X.unsqueeze(1), dim=0) / nk.unsqueeze(1)
        
        # Update covariances with better regularization
        covariances = torch.zeros_like(self.covariances)
        for k in range(self.num_components):
            if nk[k] > 1e-6:  # Only update if component has sufficient weight
                diff = X - means[k]  # [n_samples, n_features]
                weighted_diff = resp[:, k].unsqueeze(1) * diff  # [n_samples, n_features]
                cov = torch.mm(weighted_diff.T, diff) / nk[k]
                
                # Add stronger regularization for numerical stability
                reg_term = torch.eye(self.num_features, device=X.device) * max(self.reg_covar, 1e-3)
                cov += reg_term
                covariances[k] = cov
            else:
                # Fallback to identity for empty components
                covariances[k] = torch.eye(self.num_features, device=X.device)
                
        # Compute precision cholesky with better error handling
        precision_matrices = torch.zeros_like(covariances)
        precision_cholesky = torch.zeros_like(self.precision_cholesky)
        
        for k in range(self.num_components):
            try:
                # Add additional regularization if needed
                cov_reg = covariances[k] + torch.eye(self.num_features, device=X.device) * 1e-4
                precision_matrices[k] = torch.linalg.inv(cov_reg)
                precision_cholesky[k] = torch.linalg.cholesky(precision_matrices[k])
            except RuntimeError:
                # Fallback to identity if Cholesky fails
                precision_matrices[k] = torch.eye(self.num_features, device=X.device)
                precision_cholesky[k] = torch.eye(self.num_features, device=X.device)
                covariances[k] = torch.eye(self.num_features, device=X.device)
        
        return precision_cholesky, weights, means, covariances
        
    def update_params(self, weights: Tensor, means: Tensor, precision_cholesky: Tensor, covariances: Tensor, nll: Tensor):
        """Update model parameters."""
        self.weights.copy_(weights)
        self.means.copy_(means)
        self.precision_cholesky.copy_(precision_cholesky)
        self.covariances.copy_(covariances)
        self.nll.copy_(nll)
        
    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample from the fitted GMM."""
        generator = torch.Generator(device=self.weights.device)
        n_samples_comp = torch.multinomial(
            self.weights, n_samples, replacement=True, generator=generator
        ).bincount(minlength=len(self.weights))

        X = []
        
        for j, (mean, covariance, sample_count) in enumerate(
            zip(self.means, self.covariances, n_samples_comp, strict=False)
        ):
            if sample_count > 0:
                dist = torch.distributions.MultivariateNormal(mean, covariance)
                samples = dist.sample((sample_count,))
                X.append(samples)

        if X:
            return torch.vstack(X)
        else:
            return torch.randn(n_samples, self.num_features)


class FaradayModelPL(pl.LightningModule):
    """Faraday baseline: VAE + GMM composition for conditional generation.
    
    This model loads a pre-trained VAE checkpoint and fits a Gaussian Mixture Model
    in the latent space to replace the standard Gaussian prior. It maintains the
    same API as VAEModelPL for seamless integration.
    """
    
    def __init__(
        self,
        vae_checkpoint_path: str,
        train_config: TrainConfig,
        num_gmm_components: int = NUM_GMM_COMPONENTS,
        metrics_factory: Callable | None = None,
        fit_on_init: bool = True,
    ):
        super().__init__()
        
        # Load pre-trained VAE
        self.vae_model = VAEModelPL.load_from_checkpoint(vae_checkpoint_path)
        self.vae_model.eval()  # Keep VAE frozen
        
        # Freeze VAE parameters
        for param in self.vae_model.parameters():
            param.requires_grad = False
            
        # Initialize GMM
        self.gmm = GaussianMixtureModel(
            num_components=num_gmm_components,
            num_features=self.vae_model.vae.latent_dim,
        )
        
        # Store config and metadata
        self.train_config = train_config
        self.vae_checkpoint_path = vae_checkpoint_path
        self.num_gmm_components = num_gmm_components
        self.fit_on_init = fit_on_init
        
        # Copy relevant attributes from VAE for compatibility
        self.num_in_channel = self.vae_model.num_in_channel
        self.condition_dim = self.vae_model.condition_dim
        self.create_mask = self.vae_model.create_mask
        self.label_embedder = self.vae_model.label_embedder
        
        # GMM training state
        # self.gmm.initialised
        
        # Metrics
        self.metrics = metrics_factory() if metrics_factory is not None else None
        
        # Initialize dummy parameter for Lightning compatibility
        self.dummy_param = nn.Parameter(torch.tensor(0.0))
        
        self.save_hyperparameters(ignore=['vae_model'])
        
    def configure_optimizers(self):
        """Configure dummy optimizer for Lightning compatibility."""
        return torch.optim.Adam([self.dummy_param], lr=1e-6)
        
        
    def training_step(self, batch, batch_idx):
        """Batch-wise EM step for GMM training."""
        profile, condition = batch
        
        # Get condition embedding using VAE's method
        condition_emb = self.vae_model._get_condition_embedding(condition)
        
        # Encode using VAE and sample from latent distribution
        with torch.no_grad():
            profile_flat = rearrange(profile, "B L D -> B 1 (L D)")
            mu, logvar = self.vae_model.vae.encode(profile_flat, condition_emb)
            # Sample from the latent distribution
            z = self.vae_model.vae.reparameterize(mu, logvar)
        
        # Initialize GMM on first batch
        if not self.gmm.initialised:
            self.gmm.initialise(z)
            print(f"GMM initialized with batch of {z.shape[0]} samples")
            
        # Perform EM step on this batch
        log_prob_norm, log_resp = self.gmm.e_step(z)
        precision_cholesky, weights, means, covariances = self.gmm.m_step(z, log_resp)
        
        # Update GMM parameters
        nll = -log_prob_norm
        self.gmm.update_params(weights, means, precision_cholesky, covariances, nll)
        
        # Log metrics
        self.log("train/nll", nll, on_step=True, on_epoch=True, prog_bar=True)
        
        # Return dummy loss that requires gradients for Lightning compatibility
        dummy_loss = self.dummy_param * 0.0 + nll.detach()
        return dummy_loss
        
    def validation_step(self, batch, batch_idx):
        """Optional validation step to monitor GMM fitting."""
        if not self.gmm.initialised:
            return None  # cov not pos def, cannot validate

        profile, condition = batch
        condition_emb = self.vae_model._get_condition_embedding(condition)
        
        with torch.no_grad():
            profile_flat = rearrange(profile, "B L D -> B 1 (L D)")
            mu, logvar = self.vae_model.vae.encode(profile_flat, condition_emb)
            # Sample from the latent distribution for validation
            z = self.vae_model.vae.reparameterize(mu, logvar)
            
            # Compute validation NLL
            log_prob_norm, _ = self.gmm.e_step(z)
            val_nll = -log_prob_norm
            
        self.log("val/nll", val_nll, on_step=False, on_epoch=True, prog_bar=True)
        return val_nll
        
    def _format_input_y(self, *args, **kwargs):
        """Delegate to VAE's condition formatting."""
        return self.vae_model._format_input_y(*args, **kwargs)
        
    def _get_condition_embedding(self, condition: dict[str, Tensor]) -> Tensor:
        """Delegate to VAE's condition embedding."""
        return self.vae_model._get_condition_embedding(condition)
        
    @torch.no_grad()
    def sample(
        self,
        condition: dict[str, Tensor],
        num_samples: int = 1,
        return_latents: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Sample from Faraday model using GMM in latent space.
        
        Args:
            condition: Dictionary of condition tensors
            num_samples: Number of samples to generate per condition
            return_latents: Whether to return the latent codes used
            
        Returns:
            Generated samples, optionally with latent codes
        """
        self.eval()
        
        # Get condition embedding
        condition_emb = self._get_condition_embedding(condition)
        batch_size = condition_emb.shape[0]
        device = condition_emb.device
        
        # Expand conditions for multiple samples if needed
        if num_samples > 1:
            condition_emb = condition_emb.repeat_interleave(num_samples, dim=0)
            total_samples = batch_size * num_samples
        else:
            total_samples = batch_size
            
        # Sample from GMM instead of standard Gaussian
        z = self.gmm.sample(total_samples).to(device)
        
        # Decode using VAE decoder
        samples = self.vae_model.vae.decode(z, condition_emb)
        
        if return_latents:
            return samples, z
        return samples
        
    def encode_data(
        self, profile: Tensor, condition: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Encode data to latent space using VAE."""
        return self.vae_model.encode_data(profile, condition)


