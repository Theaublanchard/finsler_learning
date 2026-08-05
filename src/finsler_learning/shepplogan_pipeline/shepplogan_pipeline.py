import argparse
import torch
import logging

from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
from json import dump
from tqdm import tqdm

from torchmetrics import MeanAbsolutePercentageError

from geodesic_toolbox import (
    IdentityCoMetric,
)

from finsler_learning.logger import setup_logger
from finsler_learning.models import VAE

from finsler_learning.base_geodesic_evaluation import (
    GeodesicEvalRiemannConfig,
    GeodesicSolverParams,
)
from finsler_learning.base_quality_evaluation import (
    evaluate_dataloaders,
    eval_ae_model,
)
from finsler_learning.shepplogan_pipeline.shepplogan_utils import *
from finsler_learning.shepplogan_pipeline.shepplogan_vae import train_vae
from finsler_learning.shepplogan_pipeline.shepplogan_image_regressor import (
    get_image_regressor,
    train_regressor,
)
from finsler_learning.shepplogan_pipeline.shepplogan_randers_training import (
    train_randers,
)
from finsler_learning.shepplogan_pipeline.shepplogan_interpolation import evaluate_monotonicity
from finsler_learning.shepplogan_pipeline.shepplogan_extrapolation import (
    evaluate_extrapolation,
)
from finsler_learning.shepplogan_pipeline.shepplogan_vae_stochastic import (
    train_vae_log_var_decoder_branch,
)


##################################################################
# Evaluate VAE's impact on the regressor performances
##################################################################
@torch.no_grad()
def vae_regressor(
    vae: VAE,
    regressor: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
):
    vae.to(device)
    regressor.to(device)
    vae.eval()
    regressor.eval()
    metric_base = MeanAbsolutePercentageError().to(device)
    metric_hat = MeanAbsolutePercentageError().to(device)
    for data in tqdm(dataloader, desc="Evaluating VAE regressor", leave=False):
        x, y = data
        x = x.to(device)
        y = y.to(device)

        x_hat, _, _ = vae(x)

        preds = regressor(x).squeeze()
        preds_hat = regressor(x_hat).squeeze()

        metric_base.update(preds, y)
        metric_hat.update(preds_hat, y)
    mape_base = metric_base.compute().item()
    mape_hat = metric_hat.compute().item()
    return mape_base, mape_hat


def evaluate_vae_quality(
    export_path: Path,
    latent_dim: int,
    device: torch.device,
    batch_size: int,
):
    # We need to to do this to ensure that the VAE will not ruin
    # the performances of the regressor by being out of distribution
    # because it would ruin the goedesic evaluation later on
    model = get_models(latent_dim=latent_dim)
    try_load_pretrained_models(export_path, model)
    model = model.to(device)
    model.eval()
    regressor = get_image_regressor()
    dl_config = DatasetConfig(
        batch_size=batch_size,
    )

    shepp_logan_trainloader, shepp_logan_test_loader = get_shepplogan_dataloaders(
        dl_config,
        export_path=export_path,
    )

    mape_train_base, mape_train_hat = vae_regressor(
        vae=model,
        regressor=regressor,
        dataloader=shepp_logan_trainloader,
        device=device,
    )
    mape_test_base, mape_test_hat = vae_regressor(
        vae=model,
        regressor=regressor,
        dataloader=shepp_logan_test_loader,
        device=device,
    )

    res = {
        "mape_train_base": mape_train_base,
        "mape_train_hat": mape_train_hat,
        "mape_test_base": mape_test_base,
        "mape_test_hat": mape_test_hat,
    }
    with open(export_path / "vae_regressor_evaluation.json", "w") as f:
        dump(res, f, indent=4)
    LOGGER.info(
        f"VAE regressor evaluation results saved to {export_path / 'vae_regressor_evaluation.json'}"
    )
    LOGGER.info(f"VAE regressor evaluation results: {res}")
    LOGGER.info("Evaluating reconstruction quality...")
    encoder = model.encoder.to(device)
    decoder = model.decoder.to(device)
    encoder.eval()
    decoder.eval()
    config_euclidean = GeodesicEvalRiemannConfig(
        export_path=export_path / "geodesic_evaluation_euclidean",
        dataloader_train=shepp_logan_trainloader,
        dataloader_test=shepp_logan_test_loader,
        encoder=encoder,
        cometric=IdentityCoMetric(),
        solver_params=GeodesicSolverParams(batch_size=batch_size),
        regressor=regressor,
        decoder=decoder,
        device=device,
    )
    # Eval the reconstruction
    scores_ae = eval_ae_model(config_euclidean)
    with open(export_path / "metrics_results_ae.json", "w") as f:
        dump(scores_ae, f, indent=4)
    # Eval the train/test shift
    scores_shift = evaluate_dataloaders(
        dataloader_reference=shepp_logan_trainloader,
        dataloader_generated=shepp_logan_test_loader,
        device=device,
        train=False,
    )
    with open(export_path / "metrics_results_shift.json", "w") as f:
        dump(scores_shift, f, indent=4)


##################################################################
# All-in-one pipeline
##################################################################
def main(cfg: SheppLoganPipelineConfig):
    cfg.export_path.mkdir(parents=True, exist_ok=True)

    # Setup logger
    LOGGER = setup_logger(
        name=None,
        log_file=cfg.export_path / "pipeline.log",
        level=logging.INFO,
    )

    LOGGER.info(f"Export path: {cfg.export_path}")
    export_shepplogan_config(cfg)
    LOGGER.info("SheppLogan Pipeline Configuration:")
    LOGGER.info(f"{cfg}")

    try:
        if not cfg.skip_vae:
            LOGGER.info("Starting VAE training...")
            train_vae(
                latent_dim=cfg.latent_dim,
                export_path=cfg.export_path,
                lr=cfg.lr_vae,
                weight_decay=cfg.wd_vae,
                beta=cfg.beta,
                lbda=cfg.lbda,
                lpip_weight=cfg.lpip_weight,
                n_epochs=cfg.n_epochs_vae,
                batch_size=cfg.b_size_vae,
                label_noise=cfg.label_noise,
                device=cfg.device,
            )
            LOGGER.info("Starting VAE log_var_branch training...")
            train_vae_log_var_decoder_branch(
                latent_dim=cfg.latent_dim,
                export_path=cfg.export_path,
                lr=cfg.lr_vae,
                weight_decay=cfg.wd_vae,
                beta=cfg.beta,
                lbda=cfg.lbda,
                lpip_weight=cfg.lpip_weight,
                n_epochs=cfg.n_epochs_vae,
                batch_size=cfg.b_size_vae,
                label_noise=cfg.label_noise,
                K=cfg.K_cometric,
                kappa=1.0,
                device=cfg.device,
            )

        if not cfg.skip_regressor:
            LOGGER.info("Starting image regressor training...")
            train_regressor(
                export_path=cfg.export_path,
                batch_size=cfg.b_size_regressor,
                n_epochs=cfg.n_epochs_regressor,
                lr=cfg.lr_regressor,
                wd=cfg.wd_regressor,
                latent_dim=cfg.latent_dim,
                device=cfg.device,
            )

        if cfg.eval_regressor:
            LOGGER.info("Evaluating VAE/Regressor quality...")
            evaluate_vae_quality(
                export_path=cfg.export_path,
                latent_dim=cfg.latent_dim,
                device=cfg.device,
                batch_size=cfg.b_size_evaluation,
            )

        if not cfg.skip_randers:
            LOGGER.info("Starting Randers metric training...")
            for cometric_type in VALID_COMETRIC_TYPES:
                train_randers(
                    export_path=cfg.export_path,
                    latent_dim=cfg.latent_dim,
                    lr=cfg.lr_randers,
                    wd=cfg.wd_randers,
                    n_neighbors=cfg.n_neighbors,
                    margin=cfg.margin,
                    n_epochs=cfg.n_epochs_randers,
                    batch_size=cfg.b_size_randers,
                    use_dummy_accuracy=cfg.use_dummy_accuracy,
                    device=cfg.device,
                    cometric_type=cometric_type,
                    k=cfg.K_cometric,
                )

        if cfg.eval_interpolation or cfg.eval_quality_interpolation:
            LOGGER.info("Evaluating interpolation and/or quality...")
            evaluate_monotonicity(
                config=cfg,
                export_path=cfg.export_path,
                latent_dim=cfg.latent_dim,
                device=cfg.device,
                batch_size=cfg.b_size_evaluation,
                omega_strength=cfg.omega_strength,
                use_dummy_accuracy=cfg.use_dummy_accuracy,
                k=cfg.K_cometric,
                eval_quality=cfg.eval_quality_interpolation,
                eval_monotonicity=cfg.eval_interpolation,
            )

        if cfg.eval_extrapolation or cfg.eval_quality_extrapolation:
            LOGGER.info("Evaluating extrapolation...")
            evaluate_extrapolation(
                export_path=cfg.export_path,
                latent_dim=cfg.latent_dim,
                device=cfg.device,
                batch_size=cfg.b_size_evaluation,
                omega_strength=cfg.omega_strength,
                use_dummy_accuracy=cfg.use_dummy_accuracy,
                k=cfg.K_cometric,
                eta=cfg.eta,
                alpha=cfg.alpha,
                ema_moment=cfg.ema_moment,
                eval_quality=cfg.eval_quality_extrapolation,
                eval_extrapolation=cfg.eval_extrapolation,
                no_symmetric=cfg.no_symmetric,
                n_pts=cfg.n_pts,
            )

    except Exception as e:
        LOGGER.exception("An error occurred during the SheppLogan pipeline execution.")
        raise e


def add_global_arguments(parser: ArgumentParser):
    parser.add_argument(
        "--latent_dim", type=int, default=64, help="Dimension of the latent space"
    )
    parser.add_argument(
        "--device", type=str, default=DEVICE, help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--b_size", type=int, default=512, help="Batch size for everything except evaluation"
    )
    parser.add_argument(
        "--label_noise", type=float, default=0.0, help="Noise level for labels"
    )


def add_regressor_arguments(parser: ArgumentParser):
    parser.add_argument(
        "--n_epochs_regressor",
        type=int,
        default=150,
        help="Number of epochs for regressor training",
    )
    parser.add_argument(
        "--lr_regressor", type=float, default=1e-4, help="Learning rate for regressor"
    )
    parser.add_argument(
        "--wd_regressor", type=float, default=1e-5, help="Weight decay for regressor"
    )
    parser.add_argument(
        "--train_reg_on_vae",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train the image regressor on VAE-generated reconstructions",
    )


def add_vae_arguments(parser: ArgumentParser):
    parser.add_argument(
        "--n_epochs_vae", type=int, default=500, help="Number of epochs for VAE training"
    )
    parser.add_argument("--lr_vae", type=float, default=1e-4, help="Learning rate for VAE")
    parser.add_argument("--wd_vae", type=float, default=1e-5, help="Weight decay for VAE")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter for VAE loss")
    parser.add_argument(
        "--lbda", type=float, default=0.0, help="Lambda parameter for MS-SSIM loss"
    )
    parser.add_argument(
        "--lpip_weight",
        type=float,
        default=0.0,
        help="Weight for LPIP loss in VAE",
    )


def add_randers_arguments(parser: ArgumentParser):
    parser.add_argument(
        "--n_neighbors", type=int, default=16, help="Number of neighbors for Randers metric"
    )
    parser.add_argument("--margin", type=float, default=0, help="Margin for Randers loss")
    parser.add_argument(
        "--n_epochs_randers",
        type=int,
        default=50,
        help="Number of epochs for Randers training",
    )
    parser.add_argument(
        "--use_dummy_accuracy", action="store_true", help="Use dummy accuracy predictor"
    )
    parser.add_argument(
        "--lr_randers", type=float, default=1e-2, help="Learning rate for Randers"
    )
    parser.add_argument(
        "--wd_randers", type=float, default=1e-6, help="Weight decay for Randers"
    )


def add_evaluation_arguments(parser: ArgumentParser):
    parser.add_argument(
        "--omega_strength", type=float, default=1.0, help="Strength of omega term"
    )
    parser.add_argument(
        "--b_size_evaluation", type=int, default=64, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--n_pts", type=int, default=100, help="Number of points for geodesic evaluation"
    )


def add_extrapolation_arguments(parser: ArgumentParser):
    parser.add_argument(
        "--eta",
        type=float,
        default=0.9,
        help="Target mean cosim of the vMF samples for extrapolation",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1,
        help="Norm of the initial speed vector for extrapolation",
    )
    parser.add_argument(
        "--ema_moment",
        type=float,
        default=0.9,
        help="Exponential moving average momentum for extrapolation",
    )


def add_export_arguments(parser: ArgumentParser):
    parser.add_argument(
        "--export_path",
        type=Path,
        default=Path("/home/tblanchard/phd/finsler_learning/results/shepplogan")
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        help="Path to export results",
    )


def add_control_arguments(parser: ArgumentParser):
    parser.add_argument(
        "--skip_regressor", action="store_true", help="Skip regressor training"
    )
    parser.add_argument("--skip_vae", action="store_true", help="Skip VAE training")
    parser.add_argument(
        "--skip_latent_regressor", action="store_true", help="Skip latent regressor training"
    )
    parser.add_argument("--skip_randers", action="store_true", help="Skip Randers training")
    parser.add_argument(
        "--eval_interpolation",
        action="store_true",
        help="Evaluate interpolation",
    )
    parser.add_argument(
        "--eval_quality_interpolation",
        action="store_true",
        help="Evaluate quality metrics during interpolation",
    )
    parser.add_argument(
        "--eval_quality_extrapolation",
        action="store_true",
        help="Evaluate quality metrics during extrapolation",
    )
    parser.add_argument(
        "--eval_extrapolation",
        action="store_true",
        help="Evaluate extrapolation",
    )
    parser.add_argument(
        "--eval_regressor",
        action="store_true",
        help="Evaluate the regressor performance on the test set after training",
    )
    parser.add_argument(
        "--no_symmetric",
        action="store_true",
        help="Do not evaluate symmetric frameworks, ie only Randers-based metrics",
    )


def parse_args() -> SheppLoganPipelineConfig:
    parser = ArgumentParser(description="SheppLogan Pipeline")

    add_global_arguments(parser)
    add_regressor_arguments(parser)
    add_vae_arguments(parser)
    add_randers_arguments(parser)
    add_evaluation_arguments(parser)
    add_export_arguments(parser)
    add_control_arguments(parser)
    add_extrapolation_arguments(parser)
    args = parser.parse_args()

    # Scale lr by batch_size
    base_lr_vae = 1e-4
    base_lr_regressor = 1e-4
    base_b_size = 256
    args.lr_vae = (args.lr_vae / base_lr_vae) * (args.b_size / base_b_size) * base_lr_vae
    args.lr_regressor = (
        (args.lr_regressor / base_lr_regressor)
        * (args.b_size / base_b_size)
        * base_lr_regressor
    )

    args.export_path = Path(args.export_path).resolve()

    config = SheppLoganPipelineConfig(
        # Image regressor parameters
        n_epochs_regressor=args.n_epochs_regressor,
        b_size_regressor=args.b_size,
        lr_regressor=args.lr_regressor,
        wd_regressor=args.wd_regressor,
        # VAE parameters
        n_epochs_vae=args.n_epochs_vae,
        b_size_vae=args.b_size,
        lr_vae=args.lr_vae,
        wd_vae=args.wd_vae,
        beta=args.beta,
        lbda=args.lbda,
        lpip_weight=args.lpip_weight,
        # Randers parameters
        n_neighbors=args.n_neighbors,
        margin=args.margin,
        n_epochs_randers=args.n_epochs_randers,
        use_dummy_accuracy=args.use_dummy_accuracy,
        b_size_randers=args.b_size,
        lr_randers=args.lr_randers,
        wd_randers=args.wd_randers,
        omega_strength=args.omega_strength,
        # Evaluation parameters
        b_size_evaluation=args.b_size_evaluation,
        eta=args.eta,
        alpha=args.alpha,
        ema_moment=args.ema_moment,
        n_pts=args.n_pts,
        # Training flags
        skip_regressor=args.skip_regressor,
        skip_vae=args.skip_vae,
        skip_randers=args.skip_randers,
        # Global parameters
        eval_interpolation=args.eval_interpolation,
        eval_extrapolation=args.eval_extrapolation,
        export_path=args.export_path,
        latent_dim=args.latent_dim,
        label_noise=args.label_noise,
        device=args.device,
        eval_quality_interpolation=args.eval_quality_interpolation,
        eval_quality_extrapolation=args.eval_quality_extrapolation,
        eval_regressor=args.eval_regressor,
        no_symmetric=args.no_symmetric,
    )
    return config


if __name__ == "__main__":
    start_time = datetime.now()
    config = parse_args()
    main(cfg=config)
    end_time = datetime.now()
    duration = end_time - start_time

    # Log to the same logger that was set up in main
    LOGGER.info(f"Pipeline completed successfully!")
    LOGGER.info(f"Total execution time: {duration}")
