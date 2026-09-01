import json
import logging

from argparse import ArgumentParser
from pathlib import Path

from finsler_learning.shepplogan_pipeline.shepplogan_utils import *
from finsler_learning.logger import setup_logger
from finsler_learning.shepplogan_pipeline.shepplogan_vae_spherical import (
    train_spherical_vae,
)
from finsler_learning.shepplogan_pipeline.shepplogan_interpolation import (
    evaluate_monotonicity_spherical,
)
from finsler_learning.shepplogan_pipeline.shepplogan_extrapolation import (
    evaluate_extrapolation_spherical,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=Path,
        required=True,
        help="Path to the file shepplogan_pipeline_config.json",
    )
    return parser.parse_args()


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
            train_spherical_vae(
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

        if cfg.eval_interpolation or cfg.eval_quality_interpolation:
            LOGGER.info("Evaluating interpolation and/or quality on the hypersphere...")
            evaluate_monotonicity_spherical(
                config=cfg,
                export_path=cfg.export_path,
                latent_dim=cfg.latent_dim,
                device=cfg.device,
                batch_size=cfg.b_size_evaluation,
                eval_quality=cfg.eval_quality_interpolation,
                eval_monotonicity=cfg.eval_interpolation,
            )

        if cfg.eval_extrapolation or cfg.eval_quality_extrapolation:
            LOGGER.info("Evaluating extrapolation on the hypersphere...")
            evaluate_extrapolation_spherical(
                export_path=cfg.export_path,
                latent_dim=cfg.latent_dim,
                device=cfg.device,
                batch_size=cfg.b_size_evaluation,
                alpha=cfg.alpha,
                ema_moment=cfg.ema_moment,
                eval_quality=cfg.eval_quality_extrapolation,
                eval_extrapolation=cfg.eval_extrapolation,
                n_pts=cfg.n_pts,
            )

    except Exception as e:
        LOGGER.exception("An error occurred during the SheppLogan pipeline execution.")
        raise e


def load_config_from_json(config_path: Path) -> SheppLoganPipelineConfig:
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    cfg = SheppLoganPipelineConfig(**config_dict)
    # Enforce some types
    cfg.export_path = Path(cfg.export_path)
    cfg.device = torch.device(cfg.device)
    return cfg


def cli():
    args = parse_args()
    cfg = load_config_from_json(args.config_path)
    main(cfg)


if __name__ == "__main__":
    cli()
