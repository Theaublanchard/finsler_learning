import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor


class LGDS(torch.nn.Module):
    """Linear Gaussian Dynamical System.
    Here we use it for interpolation.
    Note that this an approximation of such models as we don't learn it using sequences of data.
    We only need (x,s) where x is the state and s is a totally ordered variable (e.g. time) to learn the parameters of the model.
    We approximate the transition matrix A using N_+ = {x_j | s_j > s_i and x_j is a neighbor of x_i} and the noise covariance Q using the residuals of the linear regression.

    A linear Gaussian dynamical system is defined by the following equations:
    x_{t+1} = A x_t + w_t where w_t ~ N(0, Q) is a Gaussian noise term.

    We learn A = argmin_A sum_{i,j in N_+} ||x_j - A x_i||^2 + reg * ||A||^2
    We learn Q as the covariance of the residuals: Q = cov({x_j - A x_i | i,j in N_+})

    Note : Performance can be poor because this implementation is not really vectorized.
    Solving the linear system in high dimension is to expensive.
    Also for the same reason we do everything on the CPU.

    Parameters:
    ----------
    T: int
        The number of time steps to interpolate between z_0 and z_T.
    d: int
        The dimension of the state space.
    k: int
        The number of neighbors to use for the fitting of the transition matrix A and the noise covariance Q.
    reg_A: float
        The regularization strength for the fitting of the transition matrix A.
    reg_Q: float
        The regularization strength for the fitting of the noise covariance Q.
    residual_metric: str
        The metric to use for computing residuals norm. Either ||.||_2 (euclidean) or ||.||_Q^-1 (q).
    """

    def __init__(
        self,
        T: int,
        d: int = 2,
        k: int = 16,
        reg_A: float = 1e-4,
        reg_Q: float = 1e-6,
        residual_metric: str = "q",
    ):
        super().__init__()
        self.T = T
        self.d = d
        self.k = k
        self.A = torch.nn.Parameter(torch.eye(d))  # Transition matrix
        self.Q = torch.nn.Parameter(torch.eye(d))  # Noise covariance

        self.reg_A = reg_A
        self.reg_Q = reg_Q
        self.residual_metric = residual_metric

        assert residual_metric.lower() in {"euclidean", "q"}, (
            "residual_metric must be one of {'euclidean', 'q'}"
        )

    def construct_n_plus(self, x: Tensor, s: Tensor):
        """Construct the set N_+ = {j | s_j > s_i and x_j is a neighbor of x_i} for each i.

        Parameters:
        ----------
        x: Tensor (n, d)
            The states of the system.
        s: Tensor (n,)
            The totally ordered variable (e.g. time) corresponding to each state.

        Returns:
        n_plus: List of Tensors
            A list of Tensors where n_plus[i] is the set N_+ for the state x_i.
        """
        x_cpu = x.to(device="cpu")
        s_cpu = s.to(device="cpu")

        knn = NearestNeighbors(
            n_neighbors=self.k + 1
        )  # +1 because the point itself is included in the neighbors
        knn.fit(x_cpu)
        indices = knn.kneighbors(x_cpu, return_distance=False)

        n_plus = []
        for i in range(x_cpu.shape[0]):
            neighbors = indices[i][1:]  # Exclude the point itself
            valid_neighbors = neighbors[
                s_cpu[neighbors] > s_cpu[i]
            ]  # Only keep neighbors with s_j > s_i
            if len(valid_neighbors) > 0:
                n_plus.append(
                    torch.tensor(valid_neighbors, dtype=torch.long, device=torch.device("cpu"))
                )  # Append valid neighbors as a tensor
            else:
                n_plus.append(
                    torch.tensor([], dtype=torch.long, device=torch.device("cpu"))
                )  # If no valid neighbors, append an empty tensor
        return n_plus

    def fit(self, x: Tensor, s: Tensor):
        """Fit the parameters of the model using the data (x,s).

        Parameters:
        ----------
        x: Tensor (n, d)
            The states of the system.
        s: Tensor (n,)
            The totally ordered variable (e.g. time) corresponding to each state.
        """
        x_cpu = x.to(device="cpu")
        s_cpu = s.to(device="cpu")

        n_plus = self.construct_n_plus(x_cpu, s_cpu)
        # Fit the transition matrix A from all (x_i -> x_j) pairs using ridge least squares.
        x_prev_list = []
        x_next_list = []
        for i in range(x_cpu.shape[0]):
            neighbors = n_plus[i]
            if len(neighbors) == 0:
                continue
            neighbors = torch.as_tensor(
                neighbors, dtype=torch.long, device=torch.device("cpu")
            )
            x_i = x_cpu[i].unsqueeze(0).expand(neighbors.shape[0], -1)
            x_j = x_cpu[neighbors]
            x_prev_list.append(x_i)
            x_next_list.append(x_j)

        if len(x_prev_list) == 0:
            raise ValueError(
                "No valid neighbor pairs found. Increase k or check ordering variable s."
            )

        X_prev = torch.cat(x_prev_list, dim=0)
        X_next = torch.cat(x_next_list, dim=0)

        d = x_cpu.shape[1]
        I = torch.eye(d, device=torch.device("cpu"), dtype=x_cpu.dtype)

        XtX = X_prev.T @ X_prev + self.reg_A * I
        YtX = X_next.T @ X_prev
        A = torch.linalg.solve(XtX, YtX.T).T
        self.A.data.copy_(A.to(device=self.A.device, dtype=self.A.dtype))

        # Compute Q as covariance-like estimate of residuals.
        residuals = X_next - X_prev @ A.T
        denom = max(residuals.shape[0] - 1, 1)
        Q = (residuals.T @ residuals) / denom
        Q = Q + self.reg_Q * I
        self.Q.data.copy_(Q.to(device=self.Q.device, dtype=self.Q.dtype))

    def _solve_block_tridiagonal(
        self, diag_blocks, lower_blocks, upper_blocks, rhs_blocks: Tensor
    ) -> Tensor:
        """Solve a block-tridiagonal linear system with a block Thomas algorithm.

        The system is
            D_t x_t + U_t x_{t+1} + L_t x_{t-1} = b_t
        with t in [0, n_blocks - 1].

        Parameters
        ----------
        diag_blocks: List[Tensor]
            Diagonal blocks D_t of shape (d, d), length n_blocks.
        lower_blocks: List[Tensor]
            Lower blocks L_t of shape (d, d), length n_blocks - 1.
            lower_blocks[t - 1] couples x_{t-1} to equation t.
        upper_blocks: List[Tensor]
            Upper blocks U_t of shape (d, d), length n_blocks - 1.
            upper_blocks[t] couples x_{t+1} to equation t.
        rhs_blocks: Tensor
            Right-hand side of shape (b, n_blocks, d).

        Returns
        -------
        Tensor
            Solution blocks x of shape (b, n_blocks, d).
        """
        b, n_blocks, d = rhs_blocks.shape

        if n_blocks == 1:
            x0 = torch.linalg.solve(diag_blocks[0], rhs_blocks[:, 0, :].T).T
            return x0.unsqueeze(1)

        c_tilde = [None] * (n_blocks - 1)
        d_tilde = [None] * n_blocks

        c_tilde[0] = torch.linalg.solve(diag_blocks[0], upper_blocks[0])
        d_tilde[0] = torch.linalg.solve(diag_blocks[0], rhs_blocks[:, 0, :].T).T

        for t in range(1, n_blocks - 1):
            denom = diag_blocks[t] - lower_blocks[t - 1] @ c_tilde[t - 1]
            c_tilde[t] = torch.linalg.solve(denom, upper_blocks[t])
            rhs_t = rhs_blocks[:, t, :] - d_tilde[t - 1] @ lower_blocks[t - 1].T
            d_tilde[t] = torch.linalg.solve(denom, rhs_t.T).T

        denom_last = diag_blocks[-1] - lower_blocks[-1] @ c_tilde[-1]
        rhs_last = rhs_blocks[:, -1, :] - d_tilde[-2] @ lower_blocks[-1].T
        d_tilde[-1] = torch.linalg.solve(denom_last, rhs_last.T).T

        x_blocks = [None] * n_blocks
        x_blocks[-1] = d_tilde[-1]

        for t in range(n_blocks - 2, -1, -1):
            x_blocks[t] = d_tilde[t] - x_blocks[t + 1] @ c_tilde[t].T

        return torch.stack(x_blocks, dim=1)

    def get_single_trajectory(self, z_0: Tensor, z_T: Tensor):
        """Compute trajectories between z_0 and z_T using the learned model.

        This solves the constrained minimization with fixed endpoints:
        sum_t ||z_{t+1} - A z_t||_W^2,
        where W is selected by residual_metric.

        Parameters
        ----------
        z_0: Tensor (b, d)
            The initial state.
        z_T: Tensor (b, d)
            The final state.

        Returns
        -------
        trajectories: Tensor (b, T, d)
            The trajectories between z_0 and z_T.
        """
        out_device = z_0.device
        out_dtype = z_0.dtype

        z_0_cpu = z_0.to(device="cpu")
        z_T_cpu = z_T.to(device="cpu")

        b, d = z_0_cpu.shape
        T = self.T

        A = self.A.to(device=torch.device("cpu"), dtype=z_0_cpu.dtype)
        At = A.T

        # Exclude endpoints z_0 and z_T.
        n_unknown = T - 2

        if n_unknown == 0:
            traj = torch.cat([z_0_cpu.unsqueeze(1), z_T_cpu.unsqueeze(1)], dim=1)
            return traj.to(device=out_device, dtype=out_dtype)

        I = torch.eye(d, device=torch.device("cpu"), dtype=z_0_cpu.dtype)
        metric_name = self.residual_metric.lower()

        if metric_name == "euclidean":
            W = I
        elif metric_name == "q":
            Q = 0.5 * (self.Q + self.Q.T)
            Q = Q.to(device=torch.device("cpu"), dtype=z_0_cpu.dtype)
            Q_reg = Q + self.reg_Q * I
            W = torch.linalg.inv(Q_reg)
        else:
            raise ValueError("residual_metric must be one of {'euclidean', 'q'}")

        # Block system: diagonal W + A^T W A, lower -W A, upper -A^T W.
        middle = W + At @ W @ A
        lower = -W @ A
        upper = -At @ W

        diag_blocks = [middle for _ in range(n_unknown)]
        lower_blocks = [lower for _ in range(n_unknown - 1)]
        upper_blocks = [upper for _ in range(n_unknown - 1)]

        rhs_blocks = torch.zeros(
            (b, n_unknown, d), device=torch.device("cpu"), dtype=z_0_cpu.dtype
        )
        rhs_blocks[:, 0, :] = ((W @ A) @ z_0_cpu.T).T
        rhs_blocks[:, -1, :] = ((At @ W) @ z_T_cpu.T).T

        z_mid = self._solve_block_tridiagonal(
            diag_blocks=diag_blocks,
            lower_blocks=lower_blocks,
            upper_blocks=upper_blocks,
            rhs_blocks=rhs_blocks,
        )

        # Concatenate full trajectory
        traj = torch.cat([z_0_cpu.unsqueeze(1), z_mid, z_T_cpu.unsqueeze(1)], dim=1)

        return traj.to(device=out_device, dtype=out_dtype)

    def get_trajectories(self, z_0: Tensor, z_T: Tensor):
        """Compute trajectories between z_0 and z_T using the learned model.

        This solves the constrained minimization with fixed endpoints:
        sum_t ||z_{t+1} - A z_t||_W^2,
        where W is selected by residual_metric.

        Parameters
        ----------
        z_0: Tensor (b, d)
            The initial state.
        z_T: Tensor (b, d)
            The final state.

        Returns
        -------
        trajectories: Tensor (b, T, d)
            The trajectories between z_0 and z_T.
        """
        trajectories = []
        for i in range(z_0.shape[0]):
            traj = self.get_single_trajectory(z_0[i : i + 1], z_T[i : i + 1])
            trajectories.append(traj)
        return torch.cat(trajectories, dim=0)

    def compute_distance(self, traj: Tensor):
        """Compute the distance of a trajectory using the learned model.

        This computes the sum of ||z_{t+1} - z_t|| along the trajectory.

        Parameters
        ----------
        traj: Tensor (b, T, d)
            The trajectories to compute the distance for.

        Returns
        -------
        distances: Tensor (b,)
            The distances of the trajectories.
        """
        b, T, d = traj.shape

        delta_pts = traj[:, 1:, :] - traj[:, :-1, :]
        delta_dst = torch.norm(delta_pts, dim=-1)
        total_distance = delta_dst.sum(dim=1)

        return total_distance
