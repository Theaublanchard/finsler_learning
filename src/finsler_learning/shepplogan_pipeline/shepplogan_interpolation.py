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


def evaluate_monotonicity_spherical(
    config: SheppLoganPipelineConfig,
    export_path: Path,
    latent_dim: int,
    device: torch.device,
    batch_size: int,
    eval_quality: bool = True,
    eval_monotonicity: bool = True,
):

    EXPORT_GEODESICS: bool = True

    model = get_spherical_model(latent_dim=latent_dim)
    dtype = next(model.parameters()).dtype
    try_load_pretrained_models(export_path / "spherical_vae", model)
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

    solver_params = GeodesicSolverParams(n_pts=config.n_pts, device=device, needs_spline=False)
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
    config_riemann = GeodesicEvalRiemannConfig(
        export_path=export_path / f"geodesic_evaluation_spherical",
        cometric=IdentityCoMetric(),
        **common_params,
    )
    if eval_monotonicity:
        geodesic_evaluation(config_riemann)
    if eval_quality:
        evaluate_framework(config_riemann)


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
    no_symmetric: bool = False,
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

    if k is None:
        k = len(shepp_logan_trainloader.dataset)

    LOGGER.info(f"Number of training samples: {len(shepp_logan_trainloader.dataset)}")
    LOGGER.info(f"Number of test samples: {len(shepp_logan_test_loader.dataset)}")

    encoder = model.encoder.to(device)
    decoder = model.decoder.to(device)
    encoder.eval()
    decoder.eval()

    solver_params = GeodesicSolverParams(n_pts=config.n_pts, device=device, needs_spline=False)
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

    # First evaluate the fast cometrics using GEORCE
    for cometric_type, cometric_info in VALID_COMETRIC_TYPES.items():
        if cometric_info["needs_spline"]:
            continue
        if cometric_info.get("is_riemann_biased", False):
            continue
        evaluate_mono_inter_setup(
            export_path,
            latent_dim,
            device,
            batch_size,
            omega_strength,
            use_dummy_accuracy,
            k,
            eval_quality,
            eval_monotonicity,
            dtype,
            common_params,
            cometric_type,
            no_symmetric,
        )

    # Then those that can't use GEORCE because way to slow to compute the cometric
    solver_params = GeodesicSolverParams(n_pts=config.n_pts, device=device, needs_spline=True)
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
    for cometric_type, cometric_info in VALID_COMETRIC_TYPES.items():
        if not cometric_info["needs_spline"]:
            continue
        if cometric_info.get("is_riemann_biased", False):
            continue
        evaluate_mono_inter_setup(
            export_path,
            latent_dim,
            device,
            batch_size,
            omega_strength,
            use_dummy_accuracy,
            k,
            eval_quality,
            eval_monotonicity,
            dtype,
            common_params,
            cometric_type,
            no_symmetric,
        )


def evaluate_mono_inter_setup(
    export_path: Path,
    latent_dim: int,
    device: torch.device,
    batch_size: int,
    omega_strength: float,
    use_dummy_accuracy: bool,
    k: int,
    eval_quality: bool,
    eval_monotonicity: bool,
    dtype: torch.dtype,
    common_params: dict,
    cometric_type: str,
    no_symmetric: bool = False,
):
    LOGGER.info(f"- Evaluating {cometric_type} metric...")
    cometric = get_cometric(
        latent_dim=latent_dim,
        export_path=export_path,
        k=k,
        batch_size=batch_size,
        cometric_type=cometric_type,
        device=device,
    ).to(dtype)
    config_riemann = GeodesicEvalRiemannConfig(
        export_path=export_path / f"geodesic_evaluation_{cometric_type}",
        cometric=cometric,
        **common_params,
    )
    if not no_symmetric:
        if eval_monotonicity:
            geodesic_evaluation(config_riemann)
        if eval_quality:
            evaluate_framework(config_riemann)

    del config_riemann

    LOGGER.info(f"- Evaluating {cometric_type} Randers metric...")
    randers = get_randers_metric_trained(
        latent_dim=latent_dim,
        cometric=cometric,
        export_path=export_path,
        beta=omega_strength,
        use_dummy_accuracy=use_dummy_accuracy,
        device=device,
        cometric_type=cometric_type,
    ).to(dtype)
    config_finsler = GeodesicEvalFinslerConfig(
        export_path=export_path / f"geodesic_evaluation_{cometric_type}_randers",
        cometric=cometric,
        randers_metric=randers,
        **common_params,
    )
    if eval_monotonicity:
        geodesic_evaluation(config_finsler)
    if eval_quality:
        evaluate_framework(config_finsler)

    del config_finsler
    del randers
