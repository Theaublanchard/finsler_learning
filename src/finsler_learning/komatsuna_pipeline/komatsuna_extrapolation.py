from finsler_learning.base_extrapolation import (
    ExtrapolationEvalFinslerConfig,
    ExtrapolationEvalRiemannConfig,
    extrapolation_evaluation,
)
from finsler_learning.base_quality_evaluation import evaluate_extrapolated_framework
from finsler_learning.komatsuna_pipeline.komatsuna_interpolation import (
    get_randers_metric_trained,
)
from finsler_learning.komatsuna_pipeline.komatsuna_nnunet import (
    get_model_regressor,
    get_vae_komatsuna_dataloaders,
)
from finsler_learning.komatsuna_pipeline.komatsuna_utils import *
from finsler_learning.komatsuna_pipeline.prepare_biased_riemann import copy_files_to_biaised


def evaluate_extrapolation(
    export_path: Path,
    latent_dim: int,
    device: torch.device,
    batch_size: int,
    omega_strength: float,
    use_dummy_accuracy: bool,
    k: int,
    eta: float = 1.0,
    alpha: float = 1.0,
    ema_moment: float = 0.9,
    eval_quality: bool = False,
    eval_extrapolation: bool = True,
    no_symmetric: bool = False,
    n_pts: int = 100,
    no_randers: bool = False,
    n_epochs_nnunet: int = 100,
) -> None:

    model = get_models(latent_dim=latent_dim)
    dtype = next(model.parameters()).dtype
    try_load_pretrained_models(export_path, model)
    model = model.to(device)
    model.eval()

    regressor = get_model_regressor(
        export_path=export_path, device=device, return_seg=False, n_epochs=n_epochs_nnunet
    )

    dl_config = DatasetConfig(batch_size=batch_size)
    komatsuna_trainloader, komatsuna_test_loader = get_komatsuna_dataloaders(
        dataset_config=dl_config, export_path=export_path
    )

    komatsuna_trainloader_vae, _ = get_vae_komatsuna_dataloaders(
        dataset_config=dl_config, latent_dim=latent_dim, device=device, export_path=export_path
    )

    LOGGER.info(f"Number of training samples: {len(komatsuna_trainloader.dataset)}")
    LOGGER.info(f"Number of test samples: {len(komatsuna_test_loader.dataset)}")

    encoder = model.encoder.to(device)
    decoder = model.decoder.to(device)
    encoder.eval()
    decoder.eval()

    common_params = {
        "latent_dim": latent_dim,
        "dataloader_train": komatsuna_trainloader,
        "dataloader_test": komatsuna_test_loader,
        "encoder": encoder,
        "decoder": decoder,
        "regressor": regressor,
        "device": device,
        "reference_dataloader_train": komatsuna_trainloader_vae,
        "alpha": alpha,
        "moving_avg_momentum": ema_moment,
        "T": n_pts,
    }

    for cometric_type, cometric_info in VALID_COMETRIC_TYPES.items():
        cometric = get_cometric(
            latent_dim=latent_dim,
            export_path=export_path,
            k=k,
            batch_size=batch_size,
            cometric_type=cometric_type,
            device=device,
        ).to(dtype)
        is_directionnal_riemann = cometric_info.get("is_riemann_biased", False)

        if is_directionnal_riemann:
            copy_files_to_biaised(export_path, cometric_type)

        if not no_symmetric and not is_directionnal_riemann:
            LOGGER.info(f"- Evaluating {cometric_type} metric...")
            config_riemann = ExtrapolationEvalRiemannConfig(
                export_path=export_path / f"extrapolation_evaluation_{cometric_type}",
                cometric=cometric,
                **common_params,
            )
            if eval_extrapolation:
                extrapolation_evaluation(config_riemann)
            if eval_quality:
                evaluate_extrapolated_framework(config_riemann)

        if not no_randers or is_directionnal_riemann:
            LOGGER.info(f"- Evaluating {cometric_type} Randers metric...")
            randers_metric = get_randers_metric_trained(
                latent_dim=latent_dim,
                cometric=cometric,
                export_path=export_path,
                beta=omega_strength,
                use_dummy_accuracy=use_dummy_accuracy,
                device=device,
                cometric_type=cometric_type,
            ).to(dtype)
            config_finsler = ExtrapolationEvalFinslerConfig(
                export_path=export_path / f"extrapolation_evaluation_{cometric_type}_randers",
                cometric=cometric,
                randers_metric=randers_metric,
                eta=eta,
                **common_params,
            )
            if eval_extrapolation:
                extrapolation_evaluation(config_finsler)
            if eval_quality:
                evaluate_extrapolated_framework(config_finsler)
            del config_finsler
            del randers_metric
            del cometric
            torch.cuda.empty_cache()


def evaluate_extrapolation_spherical(
    export_path: Path,
    latent_dim: int,
    device: torch.device,
    batch_size: int,
    alpha: float = 1.0,
    ema_moment: float = 0.9,
    eval_quality: bool = False,
    eval_extrapolation: bool = True,
    n_pts: int = 100,
    n_epochs_nnunet: int = 100,
) -> None:
    model = get_spherical_model(latent_dim=latent_dim)
    try_load_pretrained_models(export_path / "spherical_vae", model)
    model = model.to(device)
    model.eval()

    regressor = get_model_regressor(
        export_path=export_path, device=device, return_seg=False, n_epochs=n_epochs_nnunet
    )

    dl_config = DatasetConfig(batch_size=batch_size)
    komatsuna_trainloader, komatsuna_test_loader = get_komatsuna_dataloaders(
        dataset_config=dl_config, export_path=export_path
    )

    komatsuna_trainloader_vae, _ = get_vae_komatsuna_dataloaders(
        dataset_config=dl_config, latent_dim=latent_dim, device=device, export_path=export_path
    )

    LOGGER.info(f"Number of training samples: {len(komatsuna_trainloader.dataset)}")
    LOGGER.info(f"Number of test samples: {len(komatsuna_test_loader.dataset)}")

    encoder = model.encoder.to(device)
    decoder = model.decoder.to(device)
    encoder.eval()
    decoder.eval()

    common_params = {
        "latent_dim": latent_dim,
        "dataloader_train": komatsuna_trainloader,
        "dataloader_test": komatsuna_test_loader,
        "encoder": encoder,
        "decoder": decoder,
        "regressor": regressor,
        "device": device,
        "reference_dataloader_train": komatsuna_trainloader_vae,
        "alpha": alpha,
        "moving_avg_momentum": ema_moment,
        "T": n_pts,
    }

    LOGGER.info("- Evaluating spherical extrapolation...")
    config_riemann = ExtrapolationEvalRiemannConfig(
        export_path=export_path / "extrapolation_evaluation_spherical",
        cometric=IdentityCoMetric(),
        **common_params,
    )
    if eval_extrapolation:
        extrapolation_evaluation(config_riemann)
    if eval_quality:
        evaluate_extrapolated_framework(config_riemann)
