import torch
from typing import Tuple

import mash_cpp

from ma_sh.Model.base_mash import BaseMash
from ma_sh.Method.Mash.mash import toPreLoadDatas
from ma_sh.Method.rotate import toRotateVectorsFromOrthoPoses


class Mash(BaseMash):
    def __init__(
        self,
        anchor_num: int = 400,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        use_inv: bool = True,
        dtype=torch.float64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mask_boundary_sample_num: int = 90,
        sample_polar_num: int = 1000,
        sample_point_scale: float = 0.8,
    ) -> None:
        # Super Params
        self.mask_boundary_sample_num = mask_boundary_sample_num
        self.sample_polar_num = sample_polar_num
        self.sample_point_scale = sample_point_scale

        super().__init__(
            anchor_num,
            mask_degree_max,
            sh_degree_max,
            use_inv,
            dtype,
            device,
        )
        return

    def updatePreLoadDatas(self) -> bool:
        (
            self.sample_phis,
            self.sample_thetas,
            self.mask_boundary_phis,
            self.mask_boundary_phi_idxs,
            self.mask_boundary_base_values,
            self.sample_base_values,
        ) = toPreLoadDatas(
            self.anchor_num,
            self.mask_degree_max,
            self.mask_boundary_sample_num,
            self.sample_polar_num,
            self.dtype,
            self.device,
        )
        return True

    def toMaskBoundaryThetas(self) -> torch.Tensor:
        mask_boundary_thetas = mash_cpp.toMaskBoundaryThetas(
            self.mask_params,
            self.mask_boundary_base_values,
            self.mask_boundary_phi_idxs,
        )
        return mask_boundary_thetas

    def toInMaskSamplePolars(
        self, mask_boundary_thetas: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            in_mask_sample_phis,
            in_mask_sample_theta_weights,
            in_mask_sample_polar_idxs,
            in_mask_sample_base_values,
        ) = mash_cpp.toInMaskSamplePolars(
            self.anchor_num,
            self.mask_params,
            self.sample_phis,
            self.sample_thetas,
            mask_boundary_thetas,
            self.mask_boundary_phi_idxs,
            self.sample_base_values,
        )

        return (
            in_mask_sample_phis,
            in_mask_sample_theta_weights,
            in_mask_sample_polar_idxs,
            in_mask_sample_base_values,
        )

    def toSamplePointsTuple(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rotate_vectors = toRotateVectorsFromOrthoPoses(self.ortho_poses)

        (
            mask_boundary_sample_points,
            in_mask_sample_points,
            in_mask_sample_point_idxs,
        ) = mash_cpp.toMashSamplePoints(
            self.anchor_num,
            self.mask_degree_max,
            self.sh_degree_max,
            self.mask_params,
            self.sh_params,
            rotate_vectors,
            self.positions,
            self.sample_phis,
            self.sample_thetas,
            self.mask_boundary_phis,
            self.mask_boundary_phi_idxs,
            self.mask_boundary_base_values,
            self.sample_base_values,
            self.sample_point_scale,
            self.use_inv,
        )

        return (
            mask_boundary_sample_points,
            in_mask_sample_points,
            in_mask_sample_point_idxs,
        )

    def toSamplePoints(self) -> torch.Tensor:
        (
            mask_boundary_sample_points,
            in_mask_sample_points,
        ) = self.toSamplePointsTuple()[:2]

        sample_points = torch.vstack(
            [mask_boundary_sample_points, in_mask_sample_points]
        )

        return sample_points
