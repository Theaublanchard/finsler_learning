from geodesic_toolbox import (
    CoMetric,
    RandersMetrics,
)
from finsler_learning.shepplogan_pipeline.shepplogan_utils import *
from finsler_learning.base_classic_regressor import HybridRegressor
from finsler_learning.shepplogan_pipeline.shepplogan_image_regressor import (
    get_image_regressor,
    get_vae_shepplogan_dataloaders,
)
from finsler_learning.shepplogan_pipeline.shepplogan_randers_training import get_omega
from finsler_learning.base_geodesic_evaluation import (
    GeodesicSolverParams,
    GeodesicEvalFinslerConfig,
    GeodesicEvalRiemannConfig,
    geodesic_evaluation,
)
from finsler_learning.base_quality_evaluation import evaluate_framework


def get_model_regressor_trained(
    export_path: Path,
    device: torch.device,
) -> HybridRegressor:
    regressor = get_image_regressor()
    try:
        state_dict = torch.load(export_path / "regressor.pth", weights_only=True)
        regressor.load_state_dict(state_dict)
        LOGGER.info(f"Loaded pretrained regressor from {export_path / 'regressor.pth'}.")
    except FileNotFoundError:
        raise FileNotFoundError(f"No pretrained regressor found in {export_path}.")
    regressor.eval()
    regressor = regressor.to(device)
    regressor = HybridRegressor(regressor)
    return regressor


def get_randers_metric_trained(
    latent_dim: int,
    cometric: CoMetric,
    export_path: Path,
    beta: float,
    use_dummy_accuracy: bool,
    device: torch.device,
    cometric_type: str,
) -> RandersMetrics:
    omega = get_omega(latent_dim, cometric, use_dummy_accuracy=use_dummy_accuracy)
    state_dict_dir = torch.load(
        export_path / f"randers_{cometric_type}" / "direction_model.pth", weights_only=False
    )
    omega.direction.load_state_dict(state_dict_dir)
    if not use_dummy_accuracy:
        state_dict_amp = torch.load(
            export_path / f"randers_{cometric_type}" / "amplitude_model.pth",
            weights_only=False,
        )
        omega.amplitude.load_state_dict(state_dict_amp)
    omega.eval()

    randers = RandersMetrics(base_cometric=cometric, omega=omega, beta=beta)
    randers = randers.to(device)
    return randers


def evaluate_monotonicity(
    config: SheppLoganPipelineConfig,
    export_path: Path,
    latent_dim: int,
    device: torch.device,
    batch_size: int,
    omega_strength: float,
    use_dummy_accuracy: bool,
    k: int,
    eval_quality: bool = True,
    eval_monotonicity: bool = True,
):

    EXPORT_GEODESICS: bool = True

    model = get_models(latent_dim=latent_dim)
    dtype = next(model.parameters()).dtype
    try_load_pretrained_models(export_path, model)
    model = model.to(device)
    model.eval()

    regressor = get_model_regressor_trained(
        export_path=export_path,
        device=device,
    ).to(dtype)

    dl_config = DatasetConfig(batch_size=batch_size)
    shepp_logan_trainloader, shepp_logan_test_loader = get_shepplogan_dataloaders(
        dataset_config=dl_config,
        export_path=export_path,
    )
    LOGGER.info(f"Generating the images for the quality evaluation...")
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

    solver_params = GeodesicSolverParams(n_pts=config.n_pts, device=device)
    common_params = {
        "dataloader_train": shepp_logan_trainloader,
        "dataloader_test": shepp_logan_test_loader,
        "encoder": encoder,
        "regressor": regressor,
        "decoder": decoder,
        "reference_dataloader_train": shepp_logan_trainloader_vae,
        "device": device,
        "export_geodesics": EXPORT_GEODESICS,
        "solver_params": solver_params,
    }

    # # LGDS evaluation
    # LOGGER.info("- Evaluating LGDS metric...")
    # config_lgds = GeodesicEvalRiemannConfig(
    #     export_path=config.export_path / "geodesic_evaluation_lgds",
    #     cometric=IdentityCoMetric(),
    #     **common_params,
    # )
    # # Don't do it the images are too bad to have meaningful results.
    # # if eval_monotonicity:
    # #     geodesic_evaluation(config_lgds)
    # if eval_quality:
    #     evaluate_framework(config_lgds)

    # # Euclidean evaluation
    # LOGGER.info("- Evaluating Euclidean metric...")
    # config_euclidean = GeodesicEvalRiemannConfig(
    #     export_path=export_path / "geodesic_evaluation_euclidean",
    #     cometric=IdentityCoMetric(),
    #     **common_params,
    # )
    # if eval_monotonicity:
    #     geodesic_evaluation(config_euclidean)
    # if eval_quality:
    #     evaluate_framework(config_euclidean)

    # del config_euclidean

    # # Euclidean Randers evaluation
    # LOGGER.info("- Evaluating Euclidean Randers metric...")
    # randers = get_randers_metric_trained(
    #     latent_dim=latent_dim,
    #     cometric=IdentityCoMetric(),
    #     export_path=export_path,
    #     beta=omega_strength,
    #     use_dummy_accuracy=use_dummy_accuracy,
    #     device=device,
    #     cometric_type="euclidean",
    # ).to(dtype)
    # config_euclidean_randers = GeodesicEvalFinslerConfig(
    #     export_path=export_path / "geodesic_evaluation_euclidean_randers",
    #     cometric=IdentityCoMetric(),
    #     randers_metric=randers,
    #     **common_params,
    # )
    # if eval_monotonicity:
    #     geodesic_evaluation(config_euclidean_randers)
    # if eval_quality:
    #     evaluate_framework(config_euclidean_randers)

    # del config_euclidean_randers
    # del randers

    # # Riemannian VAE evaluation
    # LOGGER.info("- Evaluating Riemannian metric...")
    # cometric = get_cometric(
    #     latent_dim=latent_dim,
    #     export_path=export_path,
    #     k=k,
    #     batch_size=batch_size,
    #     cometric_type="vae",
    #     device=device,
    # ).to(dtype)
    # config_riemann_vae = GeodesicEvalRiemannConfig(
    #     export_path=export_path / "geodesic_evaluation_riemann_vae",
    #     cometric=cometric,
    #     **common_params,
    # )
    # if eval_monotonicity:
    #     geodesic_evaluation(config_riemann_vae)
    # if eval_quality:
    #     evaluate_framework(config_riemann_vae)

    # del config_riemann_vae

    # # Finsler VAE evaluation
    # LOGGER.info("- Evaluating Finsler VAE metric...")
    # randers = get_randers_metric_trained(
    #     latent_dim=latent_dim,
    #     cometric=cometric,
    #     export_path=export_path,
    #     beta=omega_strength,
    #     use_dummy_accuracy=use_dummy_accuracy,
    #     device=device,
    #     cometric_type="vae",
    # ).to(dtype)
    # config_finsler_vae = GeodesicEvalFinslerConfig(
    #     export_path=export_path / "geodesic_evaluation_finsler_vae",
    #     cometric=cometric,
    #     randers_metric=randers,
    #     **common_params,
    # )
    # if eval_monotonicity:
    #     geodesic_evaluation(config_finsler_vae)
    # if eval_quality:
    #     evaluate_framework(config_finsler_vae)

    # del config_finsler_vae
    # del cometric
    # del randers
    # torch.cuda.empty_cache()

    # # Riemannian Pullback evaluation
    # LOGGER.info("- Evaluating Riemannian Pullback metric...")
    # cometric = get_cometric(
    #     latent_dim=latent_dim,
    #     export_path=export_path,
    #     k=k,
    #     batch_size=batch_size,
    #     cometric_type="pullback",
    #     device=device,
    # ).to(dtype)
    # config_riemann_pullback = GeodesicEvalRiemannConfig(
    #     export_path=export_path / "geodesic_evaluation_riemann_pullback",
    #     cometric=cometric,
    #     **common_params,
    # )
    # if eval_monotonicity:
    #     geodesic_evaluation(config_riemann_pullback)
    # if eval_quality:
    #     evaluate_framework(config_riemann_pullback)

    # # Finsler Pullback evaluation
    # logger = logging.getLogger("shepplogan_pipeline")
    # logger.info("- Evaluating Finsler Pullback VAE metric...")
    # randers = get_randers_metric_trained(
    #     latent_dim=latent_dim,
    #     cometric=cometric,
    #     export_path=export_path,
    #     beta=omega_strength,
    #     use_dummy_accuracy=use_dummy_accuracy,
    #     device=device,
    #     cometric_type="pullback",
    # ).to(dtype)
    # config_finsler_pullback = GeodesicEvalFinslerConfig(
    #     export_path=export_path / "geodesic_evaluation_finsler_pullback",
    #     cometric=cometric,
    #     randers_metric=randers,
    #     **common_params,
    # )
    # if eval_monotonicity:
    #     geodesic_evaluation(config_finsler_pullback)
    # if eval_quality:
    #     evaluate_framework(config_finsler_pullback)

    # del config_finsler_pullback
    # del config_riemann_pullback
    # del cometric
    # del randers

    # LAND evaluation
    LOGGER.info("- Evaluating LAND metric...")
    cometric = get_cometric(
        latent_dim=latent_dim,
        export_path=export_path,
        k=k,
        batch_size=batch_size,
        cometric_type="land",
        device=device,
    ).to(dtype)
    config_land = GeodesicEvalRiemannConfig(
        export_path=export_path / "geodesic_evaluation_land",
        cometric=cometric,
        **common_params,
    )
    if eval_monotonicity:
        geodesic_evaluation(config_land)
    if eval_quality:
        evaluate_framework(config_land)

    # LAND Randers evaluation
    LOGGER.info("- Evaluating LAND Randers metric...")
    randers = get_randers_metric_trained(
        latent_dim=latent_dim,
        cometric=cometric,
        export_path=export_path,
        beta=omega_strength,
        use_dummy_accuracy=use_dummy_accuracy,
        device=device,
        cometric_type="land",
    ).to(dtype)
    config_land_randers = GeodesicEvalFinslerConfig(
        export_path=export_path / "geodesic_evaluation_land_randers",
        cometric=cometric,
        randers_metric=randers,
        **common_params,
    )
    if eval_monotonicity:
        geodesic_evaluation(config_land_randers)
    if eval_quality:
        evaluate_framework(config_land_randers)
