import torch
from torch import Tensor
from scipy.stats import kendalltau, pearsonr, spearmanr


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
        spearman_corr.append(corr)
    return torch.tensor(spearman_corr, device=x.device)