import torch
from torch import Tensor
from scipy.stats import kendalltau, pearsonr, spearmanr
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt

def hyperplane_params(pts, w, line_length=0.1):
    """Returns the points of the hyperplane defined by the normal vector w at the point pts.

    Args:
        pts: Tensor (b,2)
            The points at which the hyperplanes are defined.
        w: Tensor (b,2)
            The normal vector of the hyperplanes.
        line_length: float
            The length of the line segment of the hyperplane.

    Returns:
        pts_start: Tensor (b,2)
            The start point of the line segment of the hyperplane.
        pts_end: Tensor (b,2)
            The end point of the line segment of the hyperplane.
    """

    omega_ortho = torch.zeros_like(w)
    omega_ortho[:, 0] = -w[:, 1]
    omega_ortho[:, 1] = w[:, 0]
    omega_ortho = omega_ortho / (torch.norm(omega_ortho, dim=1, keepdim=True) + 1e-8)
    pts_start = pts - line_length / 2 * omega_ortho
    pts_end = pts + line_length / 2 * omega_ortho
    return pts_start, pts_end


def plot_hyperplane(pts, w, ax, line_length=0.1, arrow=True, title=None):
    pts_start, pts_end = hyperplane_params(pts, w, line_length=line_length)

    ax.plot(
        [pts_start[:, 0], pts_end[:, 0]],
        [pts_start[:, 1], pts_end[:, 1]],
        "r--",
    )
    if arrow:
        ax.quiver(
            pts[:, 0],
            pts[:, 1],
            w[:, 0],
            w[:, 1],
        )
    if title is not None:
        ax.set_title(title)


def kendall_tau(a: Tensor):
    """
    Measure the monotonicity of a tensor using Kendall's Tau.
    Returns 1 if the tensor is perfectly increasing, -1 if perfectly decreasing, 0 if no correlation.
    Uses a for loop and scipy's kendalltau function, which is not very efficient.

    Parameters:
    -----------
    a : Tensor (B,N)

    Returns:
    -----------
    measure : Tensor (B,)
    """
    B, n = a.shape
    kendall_tau = []
    for b in range(B):
        y = a[b].cpu().numpy()
        tau, _ = kendalltau(range(n), y)
        # If all the values are the same, kendalltau returns nan. In this case, we set tau to 0.
        if np.isnan(tau):
            tau = 0.0
        kendall_tau.append(tau)
    return torch.tensor(kendall_tau, device=a.device)


def pearson_corr(x: Tensor):
    """
    Measure the monotonicity of a tensor using Pearson correlation.
    Returns 1 if the tensor is perfectly increasing, -1 if perfectly decreasing, 0 if no correlation.

    Parameters:
    -----------
    x : Tensor (B,N)

    Returns:
    -----------
    measure : Tensor (B,)
    """
    B, n = x.shape
    pearson_corr = []
    for b in range(B):
        y = x[b].cpu().numpy()
        corr, _ = pearsonr(range(n), y)
        # If all the values are the same, pearsonr returns nan. In this case, we set corr to 0.
        if np.isnan(corr):
            corr = 0.0
        pearson_corr.append(corr)
    return torch.tensor(pearson_corr, device=x.device)

def spearman_corr(x: Tensor):
    """
    Measure the monotonicity of a tensor using Spearman correlation.
    Returns 1 if the tensor is perfectly increasing, -1 if perfectly decreasing, 0 if no correlation.

    Parameters:
    -----------
    x : Tensor (B,N)

    Returns:
    -----------
    measure : Tensor (B,)
    """
    B, n = x.shape
    spearman_corr = []
    for b in range(B):
        y = x[b].cpu().numpy()
        corr, _ = spearmanr(range(n), y)
        # If all the values are the same, spearmanr returns nan. In this case, we set corr to 0.
        if np.isnan(corr):
            corr = 0.0
        spearman_corr.append(corr)
    return torch.tensor(spearman_corr, device=x.device)

@torch.no_grad()
def decode_traj(z_traj, decoder:torch.nn.Module,device:torch.device):
    """
    Decoder a batch of latent trajectories into the original space using the provided decoder.
    
    Parameters:
    -----------
    z_traj : Tensor (B,T,D)
        The latent trajectories to decode.
    decoder : torch.nn.Module
        The decoder model to use for decoding.
    device : torch.device
        The device to use for decoding.
    """
    B, T, D = z_traj.shape
    z_traj = rearrange(z_traj, "b t d -> (b t) d")
    decoded = decoder(z_traj.to(device))
    decoded = rearrange(decoded, "(b t) ... -> b t ...", b=B, t=T)
    return decoded

def stuck_images(x_hat:torch.Tensor,pad:int=0,pad_colors=0)->torch.Tensor:
    """
    Given a sequences of images, returns the concatenated images for visualization.

    Parameters:
    x_hat : Tensor (B, T, C, H, W)
        A sequence of images.
    pad : int
        The amount of padding between images in pixels.
    pad_colors : int
        The color of the padding (0 for black, 1 for white).

    Returns:
    Tensor (B, C, H, T*W + (T-1)*pad)
        The concatenated images.
    """
    B, T, C, H, W = x_hat.shape
    if pad > 0:
        pad_tensor = torch.ones((B, T, C, H, pad), device=x_hat.device) * pad_colors
        x_hat_padded = torch.cat([x_hat, pad_tensor], dim=-1)
    else:
        x_hat_padded = x_hat
    return rearrange(x_hat_padded, "b t c h w -> b c h (t w)")


def prepare_images(x_hat:torch.Tensor, n_pts:int=10)->torch.Tensor:
    """Prepares the images for visualization by selecting a subset of points and concatenating them.
    
    Parameters:
    x_hat : Tensor (B, T, C, H, W)
        The decoded images along the geodesic trajectories.
    n_pts : int
        The number of points to select from each trajectory for visualization.
    """
    idx = torch.linspace(0, x_hat.shape[1]-1, n_pts).long()
    x_hat = x_hat[:, idx] # (B, n_pts, C, H, W)
    x_plot = stuck_images(x_hat, pad=0, pad_colors=1) # (B, C, H, T*W + (T-1)*pad)
    x_plot = rearrange(x_plot, "b c h w -> b h w c") # (B, H, W, C)
    return x_plot

def plot_results(all_res:dict, idx:int=0,sharey:bool=False,relative_variation:bool=False,randers_only:bool=False)->tuple[plt.Figure,np.ndarray]:
    """Plots the results of the geodesic evaluations for different methods.

    Parameters: 
    all_res : dict
        A dictionary containing the results for each method.
        all_res[method_name] = {
            "x_hat": Tensor (B, T, C, H, W),
            "t_traj": Tensor (B, T),
            "kendall_tau": Tensor (B,)
        }
    idx : int
        The index of the sample to plot.
    sharey : bool
        Whether to share the y-axis across subplots.
    relative_variation : bool
        Whether to display the relative variation of the t_traj
    randers_only : bool
        Whether to only plot the Randers methods.

    Returns:
    fig : matplotlib.figure.Figure
        The figure containing the plots.
    axes : numpy.ndarray
        The axes of the figure.
    """
    if randers_only:
        keep_vals = ["Randers", "Wasserstein"]
        all_res = {k: v for k, v in all_res.items() if any(keep_val in k for keep_val in keep_vals)}
    n_method = len(all_res.keys())
    fig, axes = plt.subplots(n_method, 2, figsize=(20, 2*n_method), width_ratios=[0.1, 1],
                            sharey="col" if sharey else False
                            )
    for i, (method_name, res) in enumerate(all_res.items()):
        x_hat = res["x_hat"] # (B, T, C, H, W)
        x_plot = prepare_images(x_hat, n_pts=10)
        x_plot = x_plot[idx] # (H, W, C)

        kendall_tau = res["kendall_tau"][idx]
        t_pred = res["t_traj"][idx]
        rel_val = (t_pred[-1] - t_pred[0]) / t_pred[0] * 100

        axes[i, 0].plot(t_pred, color="blue")
        axes[i, 0].text(0., 0.9, r"$\tau = {:.2f}$".format(kendall_tau), fontsize=14, ha="left", va="top", transform=axes[i, 0].transAxes)
        if relative_variation:
            axes[i, 0].text(0., 0.7, f"RV = {rel_val:.0f}%", fontsize=14, ha="left", va="top", transform=axes[i, 0].transAxes)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_ylabel(f"{method_name}",fontsize=12)
        axes[i, 1].imshow(x_plot, cmap="gray", aspect="equal")
        axes[i, 1].set_anchor("W")
        axes[i, 1].axis("off")
    fig.tight_layout()
    return fig, axes

def plot_single_traj(all_res_method:dict,idx:int=0, relative_variation:bool=False)->tuple[plt.Figure,np.ndarray]:
    """
    Plots a single trajectory for a given method.

    Parameters: 
    all_res_method : dict
        A dictionary containing the results for the method.
        all_res_method = {
            "x_hat": Tensor (B, T, C, H, W),
            "t_traj": Tensor (B, T),
            "kendall_tau": Tensor (B,)
        }
    idx : int
        The index of the sample to plot.
    relative_variation : bool
        Whether to display the relative variation of the t_traj

    Returns:
    fig : matplotlib.figure.Figure
        The figure containing the plots.
    axes : numpy.ndarray
        The axes of the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 2), width_ratios=[0.1, 1])
    x_hat = all_res_method["x_hat"] # (B, T, C, H, W)
    x_plot = prepare_images(x_hat, n_pts=10)
    x_plot = x_plot[idx] # (H, W, C)
    kendall_tau = all_res_method["kendall_tau"][idx]
    t_pred = all_res_method["t_traj"][idx]
    rel_val = (t_pred[-1] - t_pred[0]) / t_pred[0] * 100

    axes[0].plot(t_pred, color="blue")
    axes[0].text(0., 0.9, r"$\tau = {:.2f}$".format(kendall_tau), fontsize=14, ha="left", va="top", transform=axes[0].transAxes)
    if relative_variation:
        axes[0].text(0., 0.7, f"RV = {rel_val:.0f}%", fontsize=14, ha="left", va="top", transform=axes[0].transAxes)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_ylabel(f"Sequential factors",fontsize=14)
    axes[1].imshow(x_plot, cmap="gray", aspect="equal")
    axes[1].set_anchor("W")
    axes[1].axis("off")
    fig.tight_layout()
    return fig, axes
