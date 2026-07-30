from finsler_learning.base_regressor_trainer import RegressorConfig, regressor_trainer
from finsler_learning.models import ConvRegressor
from .shepplogan_utils import *


###################################################################
# Image regressor
###################################################################
def get_image_regressor() -> torch.nn.Module:
    model = ConvRegressor(in_channels=1, out_feature=1)
    return model


def get_vae_shepplogan_dataloaders(
    dataset_config: DatasetConfig,
    latent_dim: int,
    export_path: Path,
    device: torch.device,
    vae_prob: float = 1.0,
):
    # Get a dataset of Shepp-Logan reconstructed by the VAE to train the regressor on
    model = get_models(latent_dim=latent_dim)
    try_load_pretrained_models(export_path, model)
    model = model.to(device)
    model.eval()

    train_dataset = SheppLoganDataset(
        img_size=dataset_config.img_size,
        num_samples=dataset_config.num_samples_train,
        smoothing=True,
        export_path=None,
        train=True,
        normalize_area=dataset_config.normalize_area,
        label_noise=dataset_config.label_noise,
    )
    test_dataset = SheppLoganDataset(
        img_size=dataset_config.img_size,
        num_samples=dataset_config.num_samples_val,
        export_path=None,
        train=False,
        normalize_area=dataset_config.normalize_area,
        label_noise=dataset_config.label_noise,
    )

    @torch.no_grad()
    def collate_with_model(batch: list, model: VAE, device: torch.device):
        x_list = [b[0] for b in batch]
        y_list = [b[1] for b in batch]
        x_list = torch.stack(x_list, dim=0)
        y_list = torch.stack(y_list, dim=0)

        if torch.rand(1).item() < vae_prob:
            x_list = x_list.to(device)
            x_list, _, _ = model(x_list)
        return x_list.cpu(), y_list

    shepplogan_train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=True,
        num_workers=dataset_config.n_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_with_model(batch, model, device),
    )
    shepplogan_test_loader = DataLoader(
        test_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=True,
        num_workers=dataset_config.n_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_with_model(batch, model, device),
    )
    return shepplogan_train_loader, shepplogan_test_loader


def train_regressor(
    export_path: Path,
    batch_size: int,
    n_epochs: int,
    lr: float,
    wd: float,
    latent_dim: int = 32,
    device: torch.device = DEVICE,
):
    regressor = get_image_regressor()

    dl_config = DatasetConfig(
        batch_size=batch_size,
    )
    shepplogan_train_loader, shepp_logan_test_loader = get_vae_shepplogan_dataloaders(
        dataset_config=dl_config,
        latent_dim=latent_dim,
        export_path=export_path,
        device=device,
        vae_prob=0.5,  # use a mix of VAE-generated and raw images to improve robustness
    )

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(
        regressor.parameters(),
        lr=lr,
        weight_decay=wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optim,
        T_max=n_epochs,
        eta_min=1e-6,
    )
    config = RegressorConfig(
        n_epochs=n_epochs,
        dataloader=shepplogan_train_loader,
        dataloader_test=shepp_logan_test_loader,
        regressor=regressor,
        optimizer=optim,
        criterion=loss_fn,
        export_path=export_path,
        scheduler=scheduler,
    )

    regressor, loss_train, mse_train, loss_test, mse_test = regressor_trainer(config)
    return regressor, loss_train, mse_train, loss_test, mse_test
