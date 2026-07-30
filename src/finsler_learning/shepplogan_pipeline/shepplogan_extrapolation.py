from finsler_learning.shepplogan_pipeline.shepplogan_utils import *
from finsler_learning.shepplogan_pipeline.shepplogan_image_regressor import (
    get_vae_shepplogan_dataloaders,
)

from finsler_learning.base_extrapolation import (
    ExtrapolationEvalRiemannConfig,
    ExtrapolationEvalFinslerConfig,
    extrapolation_evaluation,
)
from finsler_learning.base_quality_evaluation import evaluate_extrapolated_framework
from finsler_learning.shepplogan_pipeline.shepplogan_interpolation import (
    get_model_regressor_trained,
    get_randers_metric_trained,
)


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
    no_symmetric: bool = False,
    n_pts: int = 100,
) -> None:

    model = get_models(latent_dim=latent_dim)
    dtype = next(model.parameters()).dtype
    try_load_pretrained_models(export_path, model)
    model = model.to(device)
    model.eval()

    regressor = get_model_regressor_trained(
        export_path=export_path,
        device=device,
    ).to(dtype)

    dl_config = DatasetConfig(
        batch_size=batch_size,
    )
    shepp_logan_trainloader, shepp_logan_test_loader = get_shepplogan_dataloaders(
        dataset_config=dl_config,
        export_path=export_path,
    )

    shepp_logan_trainloader_vae, _ = get_vae_shepplogan_dataloaders(
        dataset_config=dl_config,
        latent_dim=latent_dim,
        device=device,
        export_path=export_path,
    )

    LOGGER.info(f"Number of training samples: {len(shepp_logan_trainloader.dataset)}")
    LOGGER.info(f"Number of test samples: {len(shepp_logan_test_loader.dataset)}")

    encoder = model.encoder.to(device)
    decoder = model.decoder.to(device)
    encoder.eval()
    decoder.eval()

    common_params = {
        "latent_dim": latent_dim,
        "dataloader_train": shepp_logan_trainloader,
        "dataloader_test": shepp_logan_test_loader,
        "encoder": encoder,
        "decoder": decoder,
        "regressor": regressor,
        "device": device,
        "reference_dataloader_train": shepp_logan_trainloader_vae,
        "alpha": alpha,
        "moving_avg_momentum": ema_moment,
        "T": n_pts,
    }

    # # Euclidean evaluation
    # if not no_symmetric:
    #     LOGGER.info("- Evaluating Euclidean metric extrapolation...")
    #     config_euclidean = ExtrapolationEvalRiemannConfig(
    #         export_path=export_path / "extrapolation_evaluation_euclidean",
    #         cometric=IdentityCoMetric(),
    #         **common_params,
    #     )
    #     extrapolation_evaluation(config_euclidean)
    #     if eval_quality:
    #         evaluate_extrapolated_framework(config_euclidean)

    # # Euclidean Randers evaluation
    # LOGGER.info("- Evaluating Euclidean Randers metric...")
    # randers_euclidean = get_randers_metric_trained(
    #     latent_dim=latent_dim,
    #     cometric=IdentityCoMetric(),
    #     export_path=export_path,
    #     beta=omega_strength,
    #     use_dummy_accuracy=use_dummy_accuracy,
    #     device=device,
    #     cometric_type="euclidean",
    # ).to(dtype)
    # config_euclidean_randers = ExtrapolationEvalFinslerConfig(
    #     export_path=export_path / "extrapolation_evaluation_euclidean_randers",
    #     cometric=IdentityCoMetric(),
    #     randers_metric=randers_euclidean,
    #     eta=eta,
    #     **common_params,
    # )
    # extrapolation_evaluation(config_euclidean_randers)
    # if eval_quality:
    #     evaluate_extrapolated_framework(config_euclidean_randers)
    # del randers_euclidean

    # # Riemannian VAE evaluation
    # cometric_vae = get_cometric(
    #     latent_dim=latent_dim,
    #     export_path=export_path,
    #     k=k,
    #     batch_size=batch_size,
    #     cometric_type="vae",
    #     device=device,
    # ).to(dtype)
    # if not no_symmetric:
    #     LOGGER.info("- Evaluating Riemannian metric...")
    #     config_riemann_vae = ExtrapolationEvalRiemannConfig(
    #         export_path=export_path / "extrapolation_evaluation_riemann_vae",
    #         cometric=cometric_vae,
    #         **common_params,
    #     )
    #     extrapolation_evaluation(config_riemann_vae)
    #     if eval_quality:
    #         evaluate_extrapolated_framework(config_riemann_vae)

    # # Finsler VAE evaluation
    # LOGGER.info("- Evaluating Finsler VAE metric...")
    # randers_vae = get_randers_metric_trained(
    #     latent_dim=latent_dim,
    #     cometric=cometric_vae,
    #     export_path=export_path,
    #     beta=omega_strength,
    #     use_dummy_accuracy=use_dummy_accuracy,
    #     device=device,
    #     cometric_type="vae",
    # ).to(dtype)
    # config_finsler_vae = ExtrapolationEvalFinslerConfig(
    #     export_path=export_path / "extrapolation_evaluation_finsler_vae",
    #     cometric=cometric_vae,
    #     randers_metric=randers_vae,
    #     eta=eta,
    #     **common_params,
    # )
    # extrapolation_evaluation(config_finsler_vae)
    # if eval_quality:
    #     evaluate_extrapolated_framework(config_finsler_vae)
    # del config_finsler_vae
    # del randers_vae
    # del cometric_vae

    # # Riemannian Pullback evaluation
    # cometric_pullback = get_cometric(
    #     latent_dim=latent_dim,
    #     export_path=export_path,
    #     k=k,
    #     batch_size=batch_size,
    #     cometric_type="pullback",
    #     device=device,
    # ).to(dtype)
    # if not no_symmetric:
    #     LOGGER.info("- Evaluating Riemannian Pullback metric...")
    #     config_riemann_pullback = ExtrapolationEvalRiemannConfig(
    #         export_path=export_path / "extrapolation_evaluation_riemann_pullback",
    #         cometric=cometric_pullback,
    #         **common_params,
    #     )
    #     extrapolation_evaluation(config_riemann_pullback)
    #     if eval_quality:
    #         evaluate_extrapolated_framework(config_riemann_pullback)

    # # Finsler Pullback evaluation
    # LOGGER.info("- Evaluating Finsler Pullback VAE metric...")
    # randers_pullback = get_randers_metric_trained(
    #     latent_dim=latent_dim,
    #     cometric=cometric_pullback,
    #     export_path=export_path,
    #     beta=omega_strength,
    #     use_dummy_accuracy=use_dummy_accuracy,
    #     device=device,
    #     cometric_type="pullback",
    # ).to(dtype)
    # config_finsler_pullback = ExtrapolationEvalFinslerConfig(
    #     export_path=export_path / "extrapolation_evaluation_finsler_pullback",
    #     cometric=cometric_pullback,
    #     randers_metric=randers_pullback,
    #     eta=eta,
    #     **common_params,
    # )
    # extrapolation_evaluation(config_finsler_pullback)
    # if eval_quality:
    #     evaluate_extrapolated_framework(config_finsler_pullback)
    # del config_finsler_pullback
    # del randers_pullback
    # del cometric_pullback
    # torch.cuda.empty_cache()

    # LAND evaluation
    cometric_land = get_cometric(
        latent_dim=latent_dim,
        export_path=export_path,
        k=k,
        batch_size=batch_size,
        cometric_type="land",
        device=device,
    ).to(dtype)
    if not no_symmetric:
        LOGGER.info("- Evaluating LAND metric...")
        config_land = ExtrapolationEvalRiemannConfig(
            export_path=export_path / "extrapolation_evaluation_land",
            cometric=cometric_land,
            **common_params,
        )
        extrapolation_evaluation(config_land)
        if eval_quality:
            evaluate_extrapolated_framework(config_land)

    # LAND Randers evaluation
    LOGGER.info("- Evaluating LAND Randers metric...")
    randers_land = get_randers_metric_trained(
        latent_dim=latent_dim,
        cometric=cometric_land,
        export_path=export_path,
        beta=omega_strength,
        use_dummy_accuracy=use_dummy_accuracy,
        device=device,
        cometric_type="land",
    ).to(dtype)
    config_land_randers = ExtrapolationEvalFinslerConfig(
        export_path=export_path / "extrapolation_evaluation_land_randers",
        cometric=cometric_land,
        randers_metric=randers_land,
        eta=eta,
        **common_params,
    )
    extrapolation_evaluation(config_land_randers)
    if eval_quality:
        evaluate_extrapolated_framework(config_land_randers)
    del config_land_randers
    del randers_land
    del cometric_land
