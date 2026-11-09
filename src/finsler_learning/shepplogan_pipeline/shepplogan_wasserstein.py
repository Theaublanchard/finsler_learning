from argparse import ArgumentParser
import json
from pathlib import Path
import logging
from finsler_learning.base_wasserstein_learning import WassersteinConfig
from finsler_learning.base_wasserstein_learning import main as wasserstein_main
from finsler_learning.shepplogan_pipeline.shepplogan_utils import (
    SheppLoganPipelineConfig,
    get_models,
    try_load_pretrained_models,
    DatasetConfig,
    get_shepplogan_dataloaders,
)
from finsler_learning.shepplogan_pipeline.shepplogan_image_regressor import (
    get_vae_shepplogan_dataloaders,
)
from finsler_learning.shepplogan_pipeline.shepplogan_interpolation import (
    get_model_regressor_trained,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=Path,
        required=True,
        help="Path to the folder containing the pretrained model and the config",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=10,
        help="Number of bins to discretize the data",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=500,
        help="Number of epochs for potential training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for potential training",
    )
    parser.add_argument(
        "--eval_quality",
        action="store_true",
        help="Evaluate image quality of the extrapolated trajectories",
    )
    return parser.parse_args()


def load_base_config(base_path: Path) -> SheppLoganPipelineConfig:
    with open(base_path / "shepplogan_pipeline_config.json", "r") as f:
        config = json.load(f)
    config = SheppLoganPipelineConfig(**config)
    return config


def load_pretrained_models(
    config: SheppLoganPipelineConfig,
    load_quality_reference: bool = False,
):
    model = get_models(latent_dim=config.latent_dim)
    dtype = next(model.parameters()).dtype
    try_load_pretrained_models(config.export_path, model)
    model = model.to(config.device)
    model.eval()

    regressor = get_model_regressor_trained(
        export_path=config.export_path,
        device=config.device,
    ).to(dtype)

    dl_config = DatasetConfig(
        batch_size=config.b_size_evaluation,
    )
    shepp_logan_train_loader, shepp_logan_test_loader = get_shepplogan_dataloaders(
        dataset_config=dl_config,
        export_path=config.export_path,
    )

    logger.info(f"Number of training samples: {len(shepp_logan_train_loader.dataset)}")
    logger.info(f"Number of test samples: {len(shepp_logan_test_loader.dataset)}")

    encoder = model.encoder.to(config.device)
    decoder = model.decoder.to(config.device)
    encoder.eval()
    decoder.eval()
    shepp_logan_trainloader_vae = None
    if load_quality_reference:
        shepp_logan_trainloader_vae, _ = get_vae_shepplogan_dataloaders(
            dataset_config=dl_config,
            latent_dim=config.latent_dim,
            device=config.device,
            export_path=config.export_path,
        )

    return (
        encoder,
        decoder,
        regressor,
        shepp_logan_train_loader,
        shepp_logan_test_loader,
        shepp_logan_trainloader_vae,
    )


def main(args):
    config = load_base_config(args.base_path)
    config.export_path = Path(config.export_path)

    (
        encoder,
        decoder,
        regressor,
        shepp_logan_train_loader,
        shepp_logan_test_loader,
        shepp_logan_trainloader_vae,
    ) = load_pretrained_models(config, load_quality_reference=args.eval_quality)

    config_W2 = WassersteinConfig(
        export_path=config.export_path / "extrapolation_evaluation_wasserstein",
        train_loader=shepp_logan_train_loader,
        test_loader=shepp_logan_test_loader,
        encoder=encoder,
        decoder=decoder,
        regressor=regressor,
        latent_dim=config.latent_dim,
        n_bins=args.n_bins,
        lr=args.lr,
        n_epochs=args.n_epochs,
        device=config.device,
        T=config.n_pts,
        batch_size=config.b_size_evaluation,
        reference_dataloader_train=shepp_logan_trainloader_vae,
        eval_quality=args.eval_quality,
    )

    wasserstein_main(config_W2)


def cli():
    args = parse_args()
    main(args)


if __name__ == "__main__":
    cli()
