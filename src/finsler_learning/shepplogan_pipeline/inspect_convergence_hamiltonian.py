"""
This script is used to inspect the convergence of the Hamiltonian along the geodesics computed with the Finsler (and Riemannian biased)
when there is no resampling of the initial speed. 
This is demanded by Reviewer kqKM (https://openreview.net/forum?id=GflASlNXF6) albeit a bit useless imo.  
"""
import torch
import finsler_learning.shepplogan_pipeline as sp
from finsler_learning.shepplogan_pipeline import *
import finsler_learning as fl
import json
from pathlib import Path
from finsler_learning.shepplogan_pipeline.shepplogan_utils import SheppLoganPipelineConfig
from geodesic_toolbox.cometric import DualRandersMetrics
import matplotlib.pyplot as plt
from finsler_learning.utils import kendall_tau, plot_results

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
from finsler_learning.base_extrapolation import *

from geodesic_toolbox.utils import (
    sample_cometric_normal,
    sample_cone_VMF,
)


def main():
    base_path = Path(
        "/home/tblanchard/phd/finsler_learning/results/shepplogan/runs_2026-11-11"
    )
    with open(base_path / "shepplogan_pipeline_config.json", "r") as f:
        config = json.load(f)

    config = SheppLoganPipelineConfig(**config)
    config.export_path = Path(config.export_path)

    config.device = "cpu"
    latent_dim = config.latent_dim
    export_path = config.export_path
    device = config.device
    batch_size = config.b_size_evaluation
    alpha = config.alpha
    # ema_moment = config.ema_moment
    ema_moment = 1  # Do not resample
    eta = config.eta
    n_pts = config.n_pts
    k = config.n_neighbors
    omega_strength = config.omega_strength
    use_dummy_accuracy = config.use_dummy_accuracy


    compute_riemann = True



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

    cometric_type = "land"
    # cometric_info = VALID_COMETRIC_TYPES[cometric_type]

    cometric = get_cometric(
        latent_dim=latent_dim,
        export_path=export_path,
        k=k,
        batch_size=batch_size,
        cometric_type=cometric_type,
        device=device,
    ).to(dtype)
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
    if compute_riemann:
        config_riemann_biased = ExtrapolationEvalFinslerConfig(
            export_path=export_path
            / f"extrapolation_evaluation_{cometric_type}_biased_randers",
            cometric=cometric,
            randers_metric=randers_metric,
            eta=eta,
            **common_params,
        )
    x, _ = next(iter(config_finsler.dataloader_test))
    x = x.to(device)

    z,_ = config_finsler.encoder(x)
    assert not z.isnan().any()

    # Common initial speed
    v0 = get_random_exp_map(config_finsler).sample_initial_speed(z)

    N_pts_list = [25, 50, 100, 200, 500, 1000]
    H_list_finsler = []
    H_list_riemann_biased = []
    for N_pts in N_pts_list:
        print(f"Computing trajectories with {N_pts} points")
        config_finsler.T = N_pts
        exp_map_finsler = get_random_exp_map(config_finsler)
        exp_map_finsler.resample_every = N_pts + 1

        z_traj_finsler, p_traj_finsler = exp_map_finsler(z, v0)
        H_traj_finsler = exp_map_finsler.H(
            z_traj_finsler.flatten(0, 1), p_traj_finsler.flatten(0, 1)
        ).reshape(z_traj_finsler.shape[:2])
        H_list_finsler.append(H_traj_finsler.detach().cpu())

        if compute_riemann:
            config_riemann_biased.T = N_pts
            exp_map_riemann_biased = get_random_exp_map(config_riemann_biased)
            exp_map_riemann_biased.resample_every = N_pts + 1
            z_traj_riemann_biased, p_traj_riemann_biased = exp_map_riemann_biased(
                z, v0
            )
            H_traj_riemann_biased = exp_map_riemann_biased.H(
                z_traj_riemann_biased.flatten(0, 1),
                p_traj_riemann_biased.flatten(0, 1),
            ).reshape(z_traj_riemann_biased.shape[:2])
            H_list_riemann_biased.append(H_traj_riemann_biased.detach().cpu())

    if compute_riemann:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))

    for i, N_pts in enumerate(N_pts_list):
        H_f = H_list_finsler[i]
        x = torch.linspace(0, 1, H_f.shape[1])
        H0_f = H_f[:, :1]
        relative_error_f = 100 * (H_f - H0_f) / H0_f
        mean_f = relative_error_f.mean(dim=0)
        std_f = relative_error_f.std(dim=0, unbiased=False)

        line_f, = ax1.plot(x, mean_f, label=f"N_pts={N_pts}")
        ax1.fill_between(
            x, mean_f - std_f, mean_f + std_f, color=line_f.get_color(), alpha=0.2
        )

        if compute_riemann:
            H_r = H_list_riemann_biased[i]
            H0_r = H_r[:, :1]
            relative_error_r = 100 * (H_r - H0_r) / H0_r
            mean_r = relative_error_r.mean(dim=0)
            std_r = relative_error_r.std(dim=0, unbiased=False)
            line_r, = ax2.plot(x, mean_r, label=f"N_pts={N_pts}")
            ax2.fill_between(
                x, mean_r - std_r, mean_r + std_r, color=line_r.get_color(), alpha=0.2
            )

    ax1.set_title(f"Mean relative Hamiltonian evolution Finsler (+/- std, n={z.shape[0]})")
    ax1.legend()
    if compute_riemann:
        ax2.set_title(
            f"Mean relative Hamiltonian evolution Riemann (+/- std, n={z.shape[0]})"
        )
        ax2.legend()
    fig.tight_layout()
    fig.savefig(export_path / f"hamiltonian_evolution_{cometric_type}.pdf", dpi=300)


if __name__ == "__main__":
    main()