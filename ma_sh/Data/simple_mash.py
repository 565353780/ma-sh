import math
import torch
from typing import Optional

import mash_cpp

from ma_sh.Method.rotate import compute_rotation_matrix_from_ortho6d


class SimpleMash(torch.nn.Module):
    def __init__(
        self,
        anchor_num: int = 400,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        sample_phi_num: int = 10,
        sample_theta_num: int = 10,
        dtype=torch.float32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()

        # Super Params
        self.anchor_num: int = anchor_num
        self.mask_degree_max: int = mask_degree_max
        self.sh_degree_max: int = sh_degree_max
        self.sample_phi_num: int = sample_phi_num
        self.sample_theta_num: int = sample_theta_num
        self.dtype = dtype
        self.device: str = device

        self._two_pi = 2.0 * math.pi
        self._pi = math.pi

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

        self.register_buffer("sample_phis", sample_phis)
        self.register_buffer("sample_thetas", sample_thetas)

        phi_grid, theta_grid = torch.meshgrid(sample_phis, sample_thetas, indexing="ij")
        sample_phi_theta_mat = torch.stack([phi_grid, theta_grid], dim=-1)

        expanded_sample_phi_theta_mat = sample_phi_theta_mat.unsqueeze(0).expand(
            self.anchor_num, -1, -1, -1
        )
        self.register_buffer(
            "expanded_sample_phi_theta_mat", expanded_sample_phi_theta_mat
        )

        if self.mask_degree_max > 0:
            degrees = torch.arange(
                1, self.mask_degree_max + 1, dtype=self.dtype, device=self.device
            )
            angles = degrees.unsqueeze(1) * sample_phis.unsqueeze(0)
            cos_terms = torch.cos(angles)
            sin_terms = torch.sin(angles)
            self.register_buffer("cos_terms", cos_terms)
            self.register_buffer("sin_terms", sin_terms)

        self.mask_params = torch.nn.Parameter(
            torch.zeros(
                [anchor_num, 2 * mask_degree_max + 1],
                dtype=self.dtype,
                device=self.device,
            )
        )
        self.sh_params = torch.nn.Parameter(
            torch.zeros(
                [anchor_num, (sh_degree_max + 1) ** 2],
                dtype=self.dtype,
                device=self.device,
            )
        )
        self.ortho_poses = torch.nn.Parameter(
            torch.zeros([anchor_num, 6], dtype=self.dtype, device=self.device)
        )
        self.positions = torch.nn.Parameter(
            torch.zeros([anchor_num, 3], dtype=self.dtype, device=self.device)
        )

        with torch.no_grad():
            self.mask_params[:, 0] = -0.4
            self.sh_params[:, 0] = 1.0
            self.ortho_poses[:, 0] = 1.0
            self.ortho_poses[:, 4] = 1.0

    def setGradState(
        self, need_grad: bool, anchor_mask: Optional[torch.Tensor] = None
    ) -> bool:
        if anchor_mask is None:
            for param in [
                self.mask_params,
                self.sh_params,
                self.ortho_poses,
                self.positions,
            ]:
                param.requires_grad_(need_grad)
        else:
            with torch.no_grad():
                if need_grad:
                    self.mask_params[anchor_mask].requires_grad_(True)
                    self.sh_params[anchor_mask].requires_grad_(True)
                    self.ortho_poses[anchor_mask].requires_grad_(True)
                    self.positions[anchor_mask].requires_grad_(True)
                else:
                    self.mask_params[anchor_mask].requires_grad_(False)
                    self.sh_params[anchor_mask].requires_grad_(False)
                    self.ortho_poses[anchor_mask].requires_grad_(False)
                    self.positions[anchor_mask].requires_grad_(False)
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
        theta_weights = self.toMaskThetas().unsqueeze(-1)

        weighted_sample_phi_theta_mat = self.expanded_sample_phi_theta_mat.clone()
        weighted_sample_phi_theta_mat[..., 1] *= theta_weights

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

    def toSamplePoints(self) -> torch.Tensor:
        weighted_sample_phi_theta_mat = self.toWeightedSamplePhiThetaMat()

        phis, thetas = weighted_sample_phi_theta_mat.split(1, dim=-1)
        phis = phis.squeeze(-1)
        thetas = thetas.squeeze(-1)

        sample_directions = self.toDirections(phis, thetas)
        sample_distances = self.toDistances(phis, thetas)

        sample_move_vectors = sample_directions * sample_distances.unsqueeze(-1)

        rotate_mats = compute_rotation_matrix_from_ortho6d(self.ortho_poses)

        rotated_sample_move_vectors = torch.einsum(
            "b...i,bij->b...j", sample_move_vectors, rotate_mats
        )

        positions_expanded = self.positions.view(
            self.anchor_num, *((1,) * (sample_move_vectors.dim() - 2)), 3
        )
        sample_points = positions_expanded + rotated_sample_move_vectors

        return sample_points

    def forward(self) -> torch.Tensor:
        return self.toSamplePoints()
