import torch
from torch import Tensor
from tqdm import tqdm


class SphericalGeodesicSolver(torch.nn.Module):
    """Module to compute geodesics on the hypersphere.

    Parameters
    ----------
    T : int
        Number of points along the geodesic.
    """

    def __init__(self, T: int):
        super().__init__()
        self.T = T

    def get_trajectories(self, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        """Compute the geodesic between two points on the hypersphere.

        Parameters
        ----------
        z0 : torch.Tensor (B, d)
            Starting point on the hypersphere.
        z1 : torch.Tensor (B, d)
            Ending point on the hypersphere.

        Returns
        -------
        trajectory : torch.Tensor (B, T, d)
            Points along the geodesic.
        """
        # Normalize input points to ensure they are on the hypersphere
        z0 = z0 / z0.norm(dim=-1, keepdim=True)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)

        # Compute the angle between z0 and z1
        dot_product = (z0 * z1).sum(dim=-1, keepdim=True)  # (B, 1)
        theta = torch.acos(torch.clamp(dot_product, -1.0, 1.0))  # (B, 1)

        # Compute points along the geodesic using spherical linear interpolation (slerp)
        t = torch.linspace(0, 1, self.T, device=z0.device).view(1, self.T, 1)
        t = t.expand(z0.size(0), -1, -1)  # (B, T, 1)
        sin_theta = torch.sin(theta)
        sin_theta[sin_theta == 0] = 1e-8  # Avoid division by zero
        sin_theta = sin_theta[:, None, :]
        theta = theta[:, None, :]
        trajectory = (torch.sin((1 - t) * theta) / sin_theta) * z0.unsqueeze(1) + (
            torch.sin(t * theta) / sin_theta
        ) * z1.unsqueeze(1)

        return trajectory

    def compute_distance(self, traj: torch.Tensor) -> torch.Tensor:
        """Compute the geodesic distance between two points on the hypersphere.

        Parameters
        ----------
        traj : torch.Tensor (B, T, d)
            Points along the geodesic.

        Returns
        -------
        distance : torch.Tensor (B,)
            Geodesic distances between z0 and z1.
        """
        z0 = traj[:, 0, :]
        z1 = traj[:, -1, :]
        # Normalize input points to ensure they are on the hypersphere
        z0 = z0 / z0.norm(dim=-1, keepdim=True)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)

        # Compute the angle between z0 and z1
        dot_product = (z0 * z1).sum(dim=-1)  # (B,)
        theta = torch.acos(torch.clamp(dot_product, -1.0, 1.0))  # (B,)

        return theta


class RandomSphericalExpMap(torch.nn.Module):
    """Module to compute the exponential map on the hypersphere.

    Parameters
    ----------
    T_max : float
        Maximum time parameter for the exponential map.
    T : int
        Number of points along the geodesic.
    alpha : float
        Initial norm of the tangent vector.
    moving_avg_momentum : float
        Momentum for the moving average of the tangent vector.
    pbar : bool
        Whether to display a progress bar during computation.
    """

    def __init__(
        self,
        T_max: float,
        T: int,
        alpha: float,
        moving_avg_momentum: float = 0.9,
        pbar: bool = False,
    ):
        super().__init__()
        self.T_max = T_max
        self.T = T
        self.alpha = alpha
        self.moving_avg_momentum = moving_avg_momentum
        self.pbar = pbar

    def sample_initial_speed(self, q: Tensor):
        v = torch.randn_like(q)
        v = self.alpha * v / v.norm(dim=-1, keepdim=True)
        return v

    def exp_map(self, z_0: Tensor, v_0: Tensor, t: float):
        """
        Computes z_1 = exp_{z_0}(t * v_0)

        Parameters
        ----------
        z_0 : Tensor (b, d)
            Initial position on the hypersphere.
        v_0 : Tensor (b, d)
            Initial speed (tangent vector) at z_0.
        t : float
            Time parameter for the exponential map.

        Returns
        -------
        z_1 : Tensor (b, d)
            New position on the hypersphere after moving along the geodesic.
        """
        v_0_norm = v_0.norm(dim=-1, keepdim=True)
        z_1 = torch.cos(t * v_0_norm) * z_0 + torch.sin(t * v_0_norm) * (v_0 / v_0_norm)
        return z_1

    def get_trajectories(self, q0: Tensor) -> tuple[Tensor, Tensor]:
        """
        Computes the geodesic trajectory starting from position q0 and initial speed v0

        Parameters
        ----------
        q0 : torch.Tensor (b, d)
            Initial position

        Returns
        -------
        q_t : torch.Tensor (b,T,d)
            Geodesic trajectory
        p_t : torch.Tensor (b,T,d)
            Momentum along the geodesic trajectory
        """
        q_t = []
        p_t = []

        if self.pbar:
            pbar = tqdm(
                range(self.T - 1), total=self.T - 1, desc="Shooting geodesic trajectory"
            )
        else:
            pbar = range(self.T - 1)

        p0 = self.sample_initial_speed(q0)
        for t in pbar:
            p0_ = self.sample_initial_speed(q0)
            p0 = self.moving_avg_momentum * p0 + (1 - self.moving_avg_momentum) * p0_

            q_t.append(q0.detach().clone())
            p_t.append(p0.detach().clone())

            q0 = self.exp_map(q0, p0, self.T_max / (self.T - 1))

        q_t.append(q0.detach().clone())
        p_t.append(p0.detach().clone())

        q_t = torch.stack(q_t, dim=1)
        p_t = torch.stack(p_t, dim=1)

        return q_t, p_t
