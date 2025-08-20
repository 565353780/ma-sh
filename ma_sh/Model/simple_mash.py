import math
import torch

import mash_cpp

from ma_sh.Config.constant import W0
from ma_sh.Method.rotate import compute_rotation_matrix_from_ortho6d
from ma_sh.Model.base_mash import BaseMash


class SimpleMash(BaseMash):
    def __init__(
        self,
        anchor_num: int = 400,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        use_inv: bool = True,
        dtype=torch.float64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sample_phi_num: int = 10,
        sample_theta_num: int = 10,
    ) -> None:
        # Super Params
        self.sample_phi_num: int = sample_phi_num
        self.sample_theta_num: int = sample_theta_num

        self._two_pi = 2.0 * math.pi
        self._pi = math.pi

        super().__init__(
            anchor_num=anchor_num,
            mask_degree_max=mask_degree_max,
            sh_degree_max=sh_degree_max,
            use_inv=use_inv,
            dtype=dtype,
            device=device,
        )

        """
        with torch.no_grad():
            self.mask_params[:, 0] = -0.4
            self.sh_params[:, 0] = 1.0
            self.ortho_poses[:, 0] = 1.0
            self.ortho_poses[:, 4] = 1.0
        """
        return

    def updatePreLoadDatas(self) -> bool:
        sample_phis = torch.linspace(
            self._two_pi / self.sample_phi_num,
            self._two_pi,
            self.sample_phi_num,
            dtype=self.dtype,
            device=self.device,
        )
        sample_thetas = torch.linspace(
            self._pi / self.sample_theta_num,
            self._pi,
            self.sample_theta_num,
            dtype=self.dtype,
            device=self.device,
        )

        phi_grid, theta_grid = torch.meshgrid(sample_phis, sample_thetas, indexing="ij")
        sample_phi_theta_mat = torch.stack([phi_grid, theta_grid], dim=-1)

        self.expanded_sample_phi_theta_mat = sample_phi_theta_mat.unsqueeze(0).expand(
            self.anchor_num, -1, -1, -1
        )

        if self.mask_degree_max > 0:
            degrees = torch.arange(
                1, self.mask_degree_max + 1, dtype=self.dtype, device=self.device
            )
            angles = degrees.unsqueeze(1) * sample_phis.unsqueeze(0)
            self.cos_terms = torch.cos(angles)
            self.sin_terms = torch.sin(angles)
        return True

    def toMaskThetas(self) -> torch.Tensor:
        mask_thetas = self.mask_params[:, 0:1].expand(-1, self.sample_phi_num)

        if self.mask_degree_max > 0:
            cos_params = self.mask_params[:, 1::2]
            sin_params = self.mask_params[:, 2::2]

            mask_thetas = torch.addmm(mask_thetas, cos_params, self.cos_terms)
            mask_thetas = torch.addmm(mask_thetas, sin_params, self.sin_terms)

        return torch.sigmoid(mask_thetas)

    def toWeightedSamplePhiThetaMat(self) -> torch.Tensor:
        theta_weights = self.toMaskThetas()

        phis = self.expanded_sample_phi_theta_mat[..., 0]
        thetas = self.expanded_sample_phi_theta_mat[..., 1]

        weighted_thetas = thetas * theta_weights.unsqueeze(-1)

        weighted_sample_phi_theta_mat = torch.stack([phis, weighted_thetas], dim=-1)

        return weighted_sample_phi_theta_mat

    def toDirections(self, phis: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        sin_theta, cos_theta = torch.sin(thetas), torch.cos(thetas)
        sin_phi, cos_phi = torch.sin(phis), torch.cos(phis)

        x = sin_theta * cos_phi
        y = sin_theta * sin_phi
        z = cos_theta

        return torch.stack([x, y, z], dim=-1)

    def toDistances(self, phis: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        base_values = mash_cpp.toSHBaseValues(phis, thetas, self.sh_degree_max)
        base_values = base_values.transpose(1, 0)

        if base_values.dim() == 2:
            sample_distances = torch.sum(self.sh_params * base_values, dim=1)
        else:
            sample_distances = torch.einsum(
                "bn,bn...->b...", self.sh_params, base_values
            )

        return sample_distances

    def toSampleMoveVectors(self) -> torch.Tensor:
        weighted_sample_phi_theta_mat = self.toWeightedSamplePhiThetaMat()

        phis, thetas = weighted_sample_phi_theta_mat.split(1, dim=-1)
        phis = phis.squeeze(-1)
        thetas = thetas.squeeze(-1)

        sample_directions = self.toDirections(phis, thetas)
        sample_distances = self.toDistances(phis, thetas)

        sample_move_vectors = sample_directions * sample_distances.unsqueeze(-1)
        return sample_move_vectors

    def toInvSampleMoveVectors(self) -> torch.Tensor:
        sample_move_vectors = self.toSampleMoveVectors().view(self.anchor_num, -1, 3)

        inv_radius = W0 * self.sh_params[:, 0]

        inv_centers = torch.zeros(
            self.anchor_num,
            1,
            3,
            dtype=sample_move_vectors.dtype,
            device=sample_move_vectors.device,
        )
        inv_centers[:, 0, 2] = -inv_radius

        in_inv_points = sample_move_vectors - inv_centers

        in_inv_point_norms_sq = torch.sum(in_inv_points.square(), dim=2, keepdim=True)

        in_inv_point_weights = torch.rsqrt(in_inv_point_norms_sq)

        in_inv_point_directions = in_inv_points * in_inv_point_weights

        inv_radius_sq_4 = (4.0 * inv_radius.square()).view(-1, 1, 1)

        in_inv_point_lengths = inv_radius_sq_4 * in_inv_point_weights

        inv_points = torch.addcmul(
            inv_centers, in_inv_point_lengths, in_inv_point_directions
        )

        return inv_points

    def toSamplePoints(self) -> torch.Tensor:
        if self.use_inv:
            sample_move_vectors = self.toInvSampleMoveVectors()
        else:
            sample_move_vectors = self.toSampleMoveVectors()

        rotate_mats = compute_rotation_matrix_from_ortho6d(self.ortho_poses)

        rotated_sample_move_vectors = torch.einsum(
            "b...i,bij->b...j", sample_move_vectors, rotate_mats
        )

        positions_expanded = self.positions.view(
            self.anchor_num, *((1,) * (sample_move_vectors.dim() - 2)), 3
        )
        sample_points = positions_expanded + rotated_sample_move_vectors

        return sample_points
