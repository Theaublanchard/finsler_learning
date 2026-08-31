import torch
from pathlib import Path
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from finsler_learning.losses import VAELoss, MS_SSIMLoss, Losse
from finsler_learning.base_vae_trainer import vae_trainer, VAEConfig

from .komatsuna_utils import (
    DEVICE,
    LOGGER,
    get_komatsuna_dataloaders,
    get_models,
    DatasetConfig,
)


################################################################
# VAE training
################################################################
class CondVAELoss(Losse):
    def __init__(
        self,
        beta: float = 1.0,
        free_bits: float = 0.0,
        lbda: float = 0.0,
        lpip_weight: float = None,
        device=DEVICE,
        l1_weight: float = 0.0,
    ):
        super(CondVAELoss, self).__init__()
        self.lbda = lbda
        self.lpip_weight = lpip_weight
        self.l1_weight = l1_weight

        self.vae_loss = VAELoss(beta=beta, free_bits=free_bits)
        self.ms_ssim_loss = MS_SSIMLoss(
            data_range=1.0, size_average=True, n_channels=3, ndim=2
        )
        self.classif_loss = torch.nn.CrossEntropyLoss()
        if lpip_weight is not None and lpip_weight > 0.0:
            self.lpip_loss = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=True, reduction="mean"
            ).to(
                device
            )  # uses a ton of VRAM

    @property
    def beta(self) -> float:
        return self.vae_loss.beta

    @beta.setter
    def beta(self, value: float) -> None:
        self.vae_loss.beta = value

    def forward(
        self,
        x_recon,
        x,
        mu,
        log_var_latent,
        log_var_data,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Parameters:
        -----------
        x_recon : Tensor (B, C, H, W)
            Reconstructed images.
        x : Tensor (B, C, H, W)
            Original input images.
        mu : Tensor (B, latent_dim)
            Latent mean.
        log_var_latent : Tensor (B, latent_dim)
            Latent log variance.
        log_var_data : Tensor (B, ...)
            Log variance of the reconstructed data.
        return_components : bool
            If True, returns a dictionary containing the computed loss components (reconstruction loss and KL divergence).
        """
        loss_vae = self.vae_loss(x_recon, x, mu, log_var_latent, log_var_data)
        # Scale the VAE loss by image size
        loss_vae = loss_vae / x[0].numel()
        self.curr_it_component = self.vae_loss.curr_it_component
        # MS-SSIM expects values in [0, 1].
        # Use a fixed standardized clipping range to avoid batch-dependent rescaling.
        x_scaled = x.clamp(0, 1)
        x_recon_scaled = x_recon.clamp(0, 1)

        loss_ms_ssim = self.ms_ssim_loss(x_scaled, x_recon_scaled)
        self.curr_it_component = self.curr_it_component | self.ms_ssim_loss.curr_it_component
        loss = loss_vae + self.lbda * loss_ms_ssim

        if self.l1_weight is not None and self.l1_weight > 0.0:
            # force the l1 to focus on the non-white pixels, since the white pixels are not informative and can dominate the loss
            # x assumed normalized to [0, 1]
            # white pixels have all channels close to 1
            distance_from_white = 1.0 - x.mean(dim=1, keepdim=True)
            mask = distance_from_white.clamp(0, 1)
            weight = 1.0 + 10.0 * mask

            loss_l1 = (weight * (x - x_recon).abs()).mean()

            # loss_l1 = torch.nn.functional.l1_loss(x_recon, x, reduction="sum")
            # loss_l1 = (
            #     loss_l1 / x.numel()
            # )  # normalize by number of pixels to keep it on the same scale as other losses
            loss += self.l1_weight * loss_l1
            self.curr_it_component = self.curr_it_component | {"l1": loss_l1}
        if self.lpip_weight is not None and self.lpip_weight > 0.0:
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
                x_recon = x_recon.repeat(1, 3, 1, 1)
            loss_lpip = self.lpip_loss(x, x_recon)
            loss += self.lpip_weight * loss_lpip
            self.curr_it_component = self.curr_it_component | {"lpip": loss_lpip}

        self.curr_it_component = self.curr_it_component | {"total_loss": loss}
        if return_components:
            return loss, self.curr_it_component
        return loss


def train_vae(
    latent_dim: int,
    export_path: Path,
    lr: float,
    weight_decay: float,
    beta: float,
    lbda: float,
    lpip_weight: float,
    n_epochs: int,
    batch_size: int,
    l1_weight: float,
    device: torch.device = DEVICE,
) -> tuple[torch.nn.Module, list, list]:
    dl_config = DatasetConfig(
        batch_size=batch_size,
    )
    komatsuna_train_loader, komatsuna_test_loader = get_komatsuna_dataloaders(
        dataset_config=dl_config,
        export_path=export_path,
        use_augmentation=True,
    )
    model = get_models(latent_dim=latent_dim)
    # try_load_pretrained_models(export_path, model)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optim,
        T_max=n_epochs,
        eta_min=1e-6,
    )
    loss_fn = CondVAELoss(beta=beta, lbda=lbda, lpip_weight=lpip_weight,l1_weight=l1_weight)
    task_config = VAEConfig(
        n_epochs=n_epochs,
        dataloader=komatsuna_train_loader,
        dataloader_test=komatsuna_test_loader,
        model=model,
        optimizer=optim,
        criterion=loss_fn,
        export_path=export_path,
        img_shape=(3, 256, 256),
        scheduler=scheduler,
        use_beta_scheduler=True,
        device=device,
    )
    model, train_loss_list, test_loss_list = vae_trainer(task_config)
    return model, train_loss_list, test_loss_list
