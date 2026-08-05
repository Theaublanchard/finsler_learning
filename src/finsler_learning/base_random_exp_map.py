import torch
from torch import Tensor
from geodesic_toolbox.cometric import (
    CoMetric,
    RandersMetrics,
    DualRandersMetrics,
)
from geodesic_toolbox.samplers import (
    ExplicitLeapfrogIntegrator,
    ImplicitLeapfrogIntegrator,
    EulerIntegrator,
)
from geodesic_toolbox.utils import (
    sample_cometric_normal,
    sample_cone_VMF,
)
from tqdm import tqdm


class RandomExpMapRiemann(torch.nn.Module):
    """
    Shoot a geodesic like trajectory on a Riemannain metric using the shooting method.
    We use the hamiltonian formulation.
    After each integration step, the initial momentum is resampled from N(0, G(q)^{-1}) where G is the metric tensor.
    This corresponds to a random walk on the manifold where the step direction is given by the geodesic trajectory and
    the step size is given by the magnification factor of the metric at the current position.


    Parameters:
    -----------
    cometric : CoMetric
        The cometric to compute the geodesics
    T_max : float
        The maximum time to use for the geodesic trajectory.
    T: int
        The number of time points to compute along the trajectory. Step size of the integrator is computed as T_max / T.
    solver : str
        The solver to use for the shooting method. Must be one of 'leapfrog_explicit', 'leapfrog_implicit' or 'euler'. Default is 'leapfrog_implicit'.
    substep : int
        The number of substeps to use for the integrator. Default is 1 (no substeps).
    omega : float
        Coupling parameter for the explcit leapfrog integrator. Only used if solver is 'leapfrog_explicit'. Default is 10.0.
    n_fix_pts : int
        The number of fixed point iterations to use for the implicit leapfrog integrator. Only used if solver is 'leapfrog_implicit'. Default is 5.
    alpha : float
        If not None, scales the initial speed to have norm alpha.
    pbar : bool
        Whether to display a progress bar when shooting the geodesic trajectory. Default is False.
    moving_avg_momentum : float| None
        If not None, the momentum are updated with a moving average with this momentum instead of being resampled at each step. Must be between 0 and 1.
        Higher values correspond to less random trajectories. Default is None (no moving average, resample at each step).
    """

    def __init__(
        self,
        cometric: CoMetric,
        T_max=1.0,
        T=10,
        solver: str = "leapfrog_implicit",
        substeps: int = 1,
        omega: float = 10.0,
        n_fix_pts: int = 5,
        alpha: float | None = None,
        pbar: bool = False,
        moving_avg_momentum: float | None = None,
    ):
        super().__init__()
        self.cometric = cometric

        self.alpha = alpha
        self.T = T
        self.T_max = T_max
        self.t = torch.linspace(0, T_max, T)
        self.dt = self.t[1] - self.t[0]

        self.substeps = substeps
        self.solver = solver
        self.omega = omega
        self.n_fix_pts = n_fix_pts
        self.pbar = pbar
        self.moving_avg_momentum = moving_avg_momentum

        if self.solver == "leapfrog_explicit":
            self.shooter = ExplicitLeapfrogIntegrator(
                H=self.H,
                gamma=self.dt,
                omega=self.omega,
                substeps=self.substeps,
            )
        elif self.solver == "leapfrog_implicit":
            self.shooter = ImplicitLeapfrogIntegrator(
                H=self.H,
                gamma=self.dt,
                substeps=self.substeps,
                n_fix_pts=self.n_fix_pts,
            )
        elif self.solver == "euler":
            self.shooter = EulerIntegrator(
                H=self.H,
                gamma=self.dt,
                substeps=self.substeps,
            )
        else:
            raise ValueError(
                f"Unknown solver {self.solver=}. Must be one of 'leapfrog_explicit', 'leapfrog_implicit' or 'euler'."
            )

    def legendre_transform(self, q: Tensor, v: Tensor) -> Tensor:
        """
        Convert a speed vector to a momentum vector using the Legendre transform.

        Parameters
        ----------
        q : torch.Tensor (b, d)
            Position
        v : torch.Tensor (b, d)
            Speed

        Returns
        -------
        p : torch.Tensor (b, d)
            Momentum
        """
        G = self.cometric.metric_tensor(q)
        if self.cometric.is_diag:
            Gv = G * v
        else:
            Gv = torch.einsum("bij,bj->bi", G, v)
        return Gv

    def inverse_legendre_transform(self, q: Tensor, p: Tensor) -> Tensor:
        """
        Convert a momentum vector to a speed vector using the inverse Legendre transform.

        Parameters
        ----------
        q : torch.Tensor (b, d)
            Position
        p : torch.Tensor (b, d)
            Momentum

        Returns
        -------
        v : torch.Tensor (b, d)
            Speed
        """
        G_inv = self.cometric.cometric_tensor(q)
        if self.cometric.is_diag:
            G_inv_p = G_inv * p
        else:
            G_inv_p = torch.einsum("bij,bj->bi", G_inv, p)
        return G_inv_p

    def H(self, q: Tensor, p: Tensor) -> Tensor:
        """
        Hamiltonian function for the Riemannian manifold.

        Parameters
        ----------
        q : torch.Tensor (b, d)
            Position
        p : torch.Tensor (b, d)
            Momentum

        Returns
        -------
        H : torch.Tensor (b,)
            Hamiltonian value
        """
        H = self.cometric.cometric(q, p)
        return 0.5 * H

    def sample_initial_speed(self, q: Tensor) -> Tensor:
        """
        Sample an initial speed vector from N(0, G(q)^{-1}) where G is the metric tensor.

        Parameters
        ----------
        q : torch.Tensor (b, d)
            Position

        Returns
        -------
        v : torch.Tensor (b, d)
            Initial speed vector
        """
        v = sample_cometric_normal(self.cometric, q)
        if self.alpha is not None:
            v = v * self.alpha / torch.norm(v, dim=-1, keepdim=True)
        return v

    def forward(self, q0: Tensor) -> tuple[Tensor, Tensor]:
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
            pbar = tqdm(total=self.T - 1, desc="Shooting geodesic trajectory")
        else:
            pbar = range(self.T - 1)
        v0 = self.sample_initial_speed(q0)
        for _ in pbar:
            if self.moving_avg_momentum is not None:
                v0_ = self.sample_initial_speed(q0)
                v0 = self.moving_avg_momentum * v0 + (1 - self.moving_avg_momentum) * v0_
            else:
                v0 = self.sample_initial_speed(q0)
            p0 = self.legendre_transform(q0, v0)

            q_t.append(q0.detach().clone())
            p_t.append(p0.detach().clone())

            q0, p0 = self.shooter.forward(q0.detach(), p0.detach(), 2, return_traj=False)

        q_t.append(q0.detach().clone())
        p_t.append(p0.detach().clone())

        q_t = torch.stack(q_t, dim=1)  # (b,T,d)
        p_t = torch.stack(p_t, dim=1)  # (b,T,d)
        return q_t, p_t


class RandomExpMapRanders(torch.nn.Module):
    """
    Shoot a geodesic trajectory on a Randers metric using the shooting method.
    We use the hamiltonian formatuion via the dual Randers metric on the cotangent space.
    After each integration step, the initial momentum is resampled from a vMF distribution.

    Parameters:
    -----------
    randers : RandersMetric
        The base Randers metric to compute the geodesics
    T_max : float
        The maximum time to use for the geodesic trajectory.
    T: int
        The number of time points to compute along the trajectory. Step size of the integrator is computed as T_max / T.
    solver : str
        The solver to use for the shooting method. Must be one of 'leapfrog_explicit', 'leapfrog_implicit' or 'euler'. Default is 'leapfrog_implicit'.
    substep : int
        The number of substeps to use for the integrator. Default is 1 (no substeps).
    omega : float
        Coupling parameter for the explcit leapfrog integrator. Only used if solver is 'leapfrog_explicit'. Default is 10.0.
    n_fix_pts : int
        The number of fixed point iterations to use for the implicit leapfrog integrator. Only used if solver is 'leapfrog_implicit'. Default is 5.
    kappa : float
        Concentration parameter for the vMF distribution used to sample the initial momentum. Higher values correspond to less random trajectories. Default is 1.0.
    alpha : float
        Scales the initial speed to have norm alpha.
    pbar : bool
        Whether to display a progress bar when shooting the geodesic trajectory. Default is False.
    moving_avg_momentum : float| None
        If not None, the momentum are updated with a moving average with this momentum instead of being resampled at each step. Must be between 0 and 1.
        Higher values correspond to less random trajectories. Default is None (no moving average, resample at each step).
    """

    def __init__(
        self,
        randers: RandersMetrics,
        T_max=1.0,
        T=10,
        solver: str = "leapfrog_implicit",
        substeps: int = 1,
        omega: float = 10.0,
        n_fix_pts: int = 5,
        kappa: float = 1.0,
        alpha: float = 1.0,
        pbar: bool = False,
        moving_avg_momentum: float | None = None,
    ):
        super().__init__()
        self.randers = randers

        self.alpha = alpha
        self.kappa = kappa

        self.T = T
        self.T_max = T_max
        self.t = torch.linspace(0, T_max, T)
        self.dt = self.t[1] - self.t[0]

        self.substeps = substeps
        self.solver = solver
        self.omega = omega
        self.n_fix_pts = n_fix_pts
        self.pbar = pbar
        self.moving_avg_momentum = moving_avg_momentum

        self.dual_r = DualRandersMetrics(randers)

        if self.solver == "leapfrog_explicit":
            self.shooter = ExplicitLeapfrogIntegrator(
                H=self.H,
                gamma=self.dt,
                omega=self.omega,
                substeps=self.substeps,
            )
        elif self.solver == "leapfrog_implicit":
            self.shooter = ImplicitLeapfrogIntegrator(
                H=self.H,
                gamma=self.dt,
                substeps=self.substeps,
                n_fix_pts=self.n_fix_pts,
            )
        elif self.solver == "euler":
            self.shooter = EulerIntegrator(
                H=self.H,
                gamma=self.dt,
                substeps=self.substeps,
            )
        else:
            raise ValueError(
                f"Unknown solver {self.solver=}. Must be one of 'leapfrog_explicit', 'leapfrog_implicit' or 'euler'."
            )

    def legendre_transform(self, q: Tensor, v: Tensor) -> Tensor:
        """
        Convert a speed vector to a momentum vector using the Legendre transform.
        Formula given by Lemma 1.1 (iii) of 10.48550/arXiv.0808.1166 Heat flow on Finsler manifolds

        p = g(x,v) v

        Parameters
        ----------
        q : torch.Tensor (b, d)
            Position
        v : torch.Tensor (b, d)
            Speed

        Returns
        -------
        p : torch.Tensor (b, d)
            Momentum
        """
        g = self.randers.fundamental_tensor(q, v)
        p = torch.einsum("bij,bj->bi", g, v)
        return p

    def inverse_legendre_transform(self, q: Tensor, p: Tensor) -> Tensor:
        """
        Convert a momentum vector to a speed vector using the inverse Legendre transform.
        Formula given by Lemma 1.1 (iii) of 10.48550/arXiv.0808.1166 Heat flow on Finsler manifolds

        v =g^*(x,p) p

        Parameters
        ----------
        q : torch.Tensor (b, d)
            Position
        p : torch.Tensor (b, d)
            Momentum

        Returns
        -------
        v : torch.Tensor (b, d)
            Speed
        """
        g_inv = self.dual_r.fundamental_tensor(q, p)
        v = torch.einsum("bij,bj->bi", g_inv, p)
        return v

    def H(self, q: Tensor, p: Tensor) -> Tensor:
        """
        Hamiltonian function for the Riemannian manifold.

        Parameters
        ----------
        q : torch.Tensor (b, d)
            Position
        p : torch.Tensor (b, d)
            Momentum

        Returns
        -------
        H : torch.Tensor (b,)
            Hamiltonian value
        """
        H = self.dual_r(q, p)
        return 0.5 * H**2

    def forward(self, q0: Tensor) -> tuple[Tensor, Tensor]:
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
            pbar = tqdm(total=self.T - 1, desc="Shooting geodesic trajectory")
        else:
            pbar = range(self.T - 1)

        v0 = sample_cone_VMF(q0, randers=self.randers, kappa=self.kappa, alpha=self.alpha)
        for _ in pbar:
            if self.moving_avg_momentum is not None:
                v0_ = sample_cone_VMF(q0, randers=self.randers, kappa=self.kappa, alpha=self.alpha)
                v0 = self.moving_avg_momentum * v0 + (1 - self.moving_avg_momentum) * v0_
            else:
                v0 = sample_cone_VMF(q0, randers=self.randers, kappa=self.kappa, alpha=self.alpha)
            p0 = self.legendre_transform(q0, v0)

            q_t.append(q0.detach().clone())
            p_t.append(p0.detach().clone())

            q0, p0 = self.shooter.forward(q0.detach(), p0.detach(), 2, return_traj=False)

        q_t.append(q0.detach().clone())
        p_t.append(p0.detach().clone())

        q_t = torch.stack(q_t, dim=1)  # (b,T,d)
        p_t = torch.stack(p_t, dim=1)  # (b,T,d)

        return q_t, p_t
