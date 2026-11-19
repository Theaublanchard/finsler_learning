import torch
import kmedoids


class VarGP(torch.nn.Module):
    def __init__(self, centroids: torch.Tensor, K: int = None, tau: float = None):
        super().__init__()

        if centroids is not None:
            self.register_buffer("centroids", centroids)

        if K is not None and centroids is not None:
            self.process_centroids(K)
        elif K is None and centroids is not None:
            self.K = self.centroids.size(0)
        else:
            self.K = K

        if tau is None:
            tau_ = self.get_tau()
        else:
            tau_ = torch.tensor(tau, dtype=centroids.dtype, device=centroids.device)

        self.register_buffer("tau", tau_)
        self.register_buffer("inv_K_zz_reg", self.init_kernel())

    def process_centroids(self, K: int) -> None:
        """
        Process the centroids to select K representative centroids using K-Medoids clustering.

        Parameters:
        K : int
            The number of centroids to select. If K=-1, use all centroids.
        """
        if K <= self.centroids.shape[0] and K > 0:
            self.K = K
            dst_mat = torch.cdist(self.centroids, self.centroids, p=2).sqrt().cpu().numpy()
            kmedoids_model = kmedoids.KMedoids(
                n_clusters=K, metric="precomputed", random_state=1312
            )
            kmedoids_model.fit(dst_mat)
            centroids_idx = kmedoids_model.medoid_indices_

            self.centroids = self.centroids[centroids_idx]
        elif K == -1:
            self.K = self.centroids.shape[0]
        else:
            print(
                f"Warning: K={K} is greater than the number of centroids {self.centroids.shape[0]}. Using all centroids."
            )
            self.K = self.centroids.shape[0]

    def get_tau(self) -> torch.Tensor:
        """
        Set the temperature to the maximum minimum distance between centroids scaled by temperature_scale.
        """
        dst_mat = torch.cdist(self.centroids, self.centroids, p=2)
        dst_mat[dst_mat == 0] = float("inf")  # Avoid zero self distances
        # min_distances, _ = dst_mat.min(dim=1)
        # self.temperature = min_distances.max()
        # Find distance to second closest centroid
        sorted_distances, _ = torch.sort(dst_mat, dim=1)
        second_min_distances = sorted_distances[:, 1]
        return second_min_distances.max()

    def kernel(self, x: torch.Tensor, y: torch.Tensor, tau):
        """
        RBF kernel function.

        Parameters
        ----------
        x : torch.Tensor (N, d)
            First input tensor.
        y : torch.Tensor (M, d)
            Second input tensor.
        tau : float
            Length scale parameter.
        """
        return torch.exp(-torch.sum((x[:, None] - y[None, :]) ** 2, dim=-1) / (2 * tau**2))

    def init_kernel(self) -> torch.Tensor:
        K_zz = self.kernel(self.centroids, self.centroids, self.tau)  # (K, K)
        K_zz_reg = K_zz + self.tau * torch.eye(
            K_zz.size(0), device=K_zz.device, dtype=K_zz.dtype
        )
        inv_K_zz_reg = torch.linalg.inv(K_zz_reg)
        return inv_K_zz_reg

    def s(self, z):
        """
        Compute the variance of the GP at the input points z.

        Parameters
        ----------
        z : torch.Tensor (N, d)
            Input tensor.

        Returns
        -------
        torch.Tensor (N)
            The computed s(z) values for each input.
        """
        K_zz = self.kernel(z, z, self.tau)  # (N, N)
        K_zZ = self.kernel(z, self.centroids, self.tau)  # (N, K)
        s_z = K_zz - K_zZ @ self.inv_K_zz_reg @ K_zZ.T
        return s_z.diagonal()

    def forward(self, z):
        """
        Compute the variance of the GP at the input points z.

        Parameters
        ----------
        z : torch.Tensor (N, d)
            Input tensor.

        Returns
        -------
        torch.Tensor (N)
            The computed s(z) values for each input.
        """
        return self.s(z)


class WarpedDecoder(torch.nn.Module):
    def __init__(
        self,
        decoder: torch.nn.Module,
        centroids: torch.Tensor,
        beta: float = 1.0,
        K: int = None,
        tau: float = None,
    ):
        super().__init__()
        self.decoder = decoder
        self.s = VarGP(centroids=centroids, K=K, tau=tau)
        self.beta = beta

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the warped decoder.

        Parameters
        ----------
        z : torch.Tensor (B, d)
            Input tensor.

        Returns
        -------
        torch.Tensor (B, output_dim)
            The output of the warped decoder. Here the output size
            is the product of the decoder output size + 1
        """
        s_z = self.s(z)  # (B,)
        x_hat = self.decoder(z)
        x_hat_flat = x_hat.flatten(start_dim=1)  # (B, output_dim_dec)
        warped_output = torch.cat(
            [x_hat_flat, self.beta * s_z.unsqueeze(1)], dim=1
        )  # (B, output_dim_dec + 1)
        return warped_output

class FlatDecoderWrapper(torch.nn.Module):
    def __init__(
        self,
        decoder: torch.nn.Module,
    ):
        super().__init__()
        self.decoder = decoder

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the warped decoder.

        Parameters
        ----------
        z : torch.Tensor (B, d)
            Input tensor.

        Returns
        -------
        torch.Tensor (B, output_dim)
            The output of the warped decoder. Here the output size
            is the product of the decoder output size + 1
        """
        x_hat = self.decoder(z)
        x_hat_flat = x_hat.flatten(start_dim=1)  # (B, output_dim_dec)
        return x_hat_flat