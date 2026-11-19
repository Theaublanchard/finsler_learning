import json
import logging
from argparse import ArgumentParser
from pathlib import Path

from finsler_learning.komatsuna_pipeline.komatsuna_extrapolation import (
    evaluate_extrapolation_spherical,
)
from finsler_learning.komatsuna_pipeline.komatsuna_interpolation import (
    evaluate_monotonicity_spherical,
)
from finsler_learning.komatsuna_pipeline.komatsuna_utils import *
from finsler_learning.komatsuna_pipeline.komatsuna_vae_spherical import train_spherical_vae
from finsler_learning.logger import setup_logger


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=Path,
        required=True,
        help="Path to the folder containing the komatsuna_pipeline_config.json",
    )
    parser.add_argument(
        "--skip_vae",
        action="store_true",
        help="Skip training the VAE and use the pretrained model",
    )
    parser.add_argument(
        "--eval_interpolation",
        action="store_true",
        help="Evaluate interpolation on the hypersphere",
    )
    parser.add_argument(
        "--eval_quality_interpolation",
        action="store_true",
        help="Evaluate quality of interpolation on the hypersphere",
    )
    parser.add_argument(
        "--eval_extrapolation",
        action="store_true",
        help="Evaluate extrapolation on the hypersphere",
    )
    parser.add_argument(
        "--eval_quality_extrapolation",
        action="store_true",
        help="Evaluate quality of extrapolation on the hypersphere",
    )
    return parser.parse_args()


def main(cfg: KomatsunaPipelineConfig):
    cfg.export_path.mkdir(parents=True, exist_ok=True)

    # Setup logger
    LOGGER = setup_logger(
        name=None, log_file=cfg.export_path / "pipeline.log", level=logging.INFO
    )

    LOGGER.info(f"Export path: {cfg.export_path}")
    export_komatsuna_config(cfg)
    LOGGER.info("Komatsuna Pipeline Configuration:")
    LOGGER.info(f"{cfg}")

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
            l1_weight=cfg.l1_weight,
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
            n_epochs_nnunet=cfg.n_epochs_nnunet,
        )


def load_config_from_json(args) -> KomatsunaPipelineConfig:
    config_path = Path(args.config_path).resolve() / "komatsuna_pipeline_config.json"
    print(f"Resolved config path: {config_path}")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    cfg = KomatsunaPipelineConfig(**config_dict)
    # Enforce some types
    cfg.export_path = Path(cfg.export_path)
    cfg.device = torch.device(cfg.device)
    cfg.skip_vae = args.skip_vae
    cfg.eval_quality_interpolation = args.eval_quality_interpolation
    cfg.eval_quality_extrapolation = args.eval_quality_extrapolation
    cfg.eval_interpolation = args.eval_interpolation
    cfg.eval_extrapolation = args.eval_extrapolation
    return cfg


def cli():
    args = parse_args()
    cfg = load_config_from_json(args)
    print("Loader coonfig", cfg)
    main(cfg)


if __name__ == "__main__":
    cli()
