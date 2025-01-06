import torch
import numpy as np
import open3d as o3d
from math import sqrt
from torch.autograd import grad
from typing import Union, Tuple

import mash_cpp

from ma_sh.Model.base_mash import BaseMash
from ma_sh.Method.data import toNumpy
from ma_sh.Method.normal import toNormalTags
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Method.Mash.mash import toPreLoadDatas


class Mash(BaseMash):
    def __init__(
        self,
        anchor_num: int = 400,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        mask_boundary_sample_num: int = 90,
        sample_polar_num: int = 1000,
        sample_point_scale: float = 0.8,
        use_inv: bool = True,
        idx_dtype=torch.int64,
        dtype=torch.float64,
        device: str = "cpu",
    ) -> None:
        # Super Params
        self.mask_boundary_sample_num = mask_boundary_sample_num
        self.sample_polar_num = sample_polar_num
        self.sample_point_scale = sample_point_scale

        # Pre Load Datas
        self.sample_thetas = torch.tensor([0.0], dtype=dtype).to(device)
        self.mask_boundary_phis = torch.tensor([0.0], dtype=dtype).to(device)
        self.mask_boundary_base_values = torch.tensor([0.0], dtype=dtype).to(device)

        super().__init__(
            anchor_num,
            mask_degree_max,
            sh_degree_max,
            use_inv,
            idx_dtype,
            dtype,
            device,
        )
        return

    @classmethod
    def fromMash(
        cls,
        target_mash,
        anchor_num: Union[int, None] = None,
        mask_degree_max: Union[int, None] = None,
        sh_degree_max: Union[int, None] = None,
        mask_boundary_sample_num: Union[int, None] = None,
        sample_polar_num: Union[int, None] = None,
        sample_point_scale: Union[float, None] = None,
        use_inv: Union[bool, None] = None,
        idx_dtype=None,
        dtype=None,
        device: Union[str, None] = None,
        ):

        mash = Mash(
            anchor_num if anchor_num is not None else target_mash.anchor_num,
            mask_degree_max if mask_degree_max is not None else target_mash.mask_degree_max,
            sh_degree_max if sh_degree_max is not None else target_mash.sh_degree_max,
            mask_boundary_sample_num if mask_boundary_sample_num is not None else target_mash.mask_boundary_sample_num,
            sample_polar_num if sample_polar_num is not None else target_mash.sample_polar_num,
            sample_point_scale if sample_point_scale is not None else target_mash.sample_point_scale,
            use_inv if use_inv is not None else target_mash.use_inv,
            idx_dtype if idx_dtype is not None else target_mash.idx_dtype,
            dtype if dtype is not None else target_mash.dtype,
            device if device is not None else target_mash.device,
        )
        return mash

    @classmethod
    def fromParamsDict(
        cls,
        params_dict: dict,
        mask_boundary_sample_num: int = 90,
        sample_polar_num: int = 1000,
        sample_point_scale: float = 0.8,
        idx_dtype=torch.int64,
        dtype=torch.float64,
        device: str = "cpu",
    ):
        mask_params = params_dict["mask_params"]
        sh_params = params_dict["sh_params"]
        use_inv = params_dict["use_inv"]

        anchor_num = mask_params.shape[0]
        mask_degree_max = int((mask_params.shape[1] - 1) / 2)
        sh_degree_max = int(sqrt(sh_params.shape[1] - 1))

        mash = cls(
            anchor_num,
            mask_degree_max,
            sh_degree_max,
            mask_boundary_sample_num,
            sample_polar_num,
            sample_point_scale,
            use_inv,
            idx_dtype,
            dtype,
            device,
        )

        mash.loadParamsDict(params_dict)

        return mash

    @classmethod
    def fromParamsFile(
        cls,
        params_file_path: str,
        mask_boundary_sample_num: int = 90,
        sample_polar_num: int = 1000,
        sample_point_scale: float = 0.8,
        idx_dtype=torch.int64,
        dtype=torch.float64,
        device: str = "cpu",
    ):
        params_dict = np.load(params_file_path, allow_pickle=True).item()

        return cls.fromParamsDict(
            params_dict,
            mask_boundary_sample_num,
            sample_polar_num,
            sample_point_scale,
            idx_dtype,
            dtype,
            device,
        )

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
            self.idx_dtype,
            self.dtype,
            self.device,
        )
        return True

    def toMaskBoundaryThetas(self) -> torch.Tensor:
        mask_boundary_thetas = mash_cpp.toMaskBoundaryThetas(self.mask_params, self.mask_boundary_base_values, self.mask_boundary_phi_idxs)
        return mask_boundary_thetas

    def toInMaskSamplePolars(self, mask_boundary_thetas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            in_mask_sample_phis,
            in_mask_sample_theta_weights,
            in_mask_sample_polar_idxs,
            in_mask_sample_base_values
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
            in_mask_sample_base_values
        )

    def toSamplePointsWithNormals(self, refine_normals: bool=False, fps_sample_scale: float = -1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.mask_boundary_phis.requires_grad_(True)

        mask_boundary_thetas = self.toMaskBoundaryThetas()

        mask_boundary_thetas.requires_grad_(True)

        (
            in_mask_sample_phis,
            in_mask_sample_theta_weights,
            in_mask_sample_polar_idxs,
            in_mask_sample_base_values
        ) = self.toInMaskSamplePolars(mask_boundary_thetas)

        in_mask_sample_phis.requires_grad_(True)
        in_mask_sample_theta_weights.requires_grad_(True)

        in_mask_sh_points = self.toWeightedSamplePoints(in_mask_sample_phis, in_mask_sample_theta_weights,
            in_mask_sample_polar_idxs, in_mask_sample_base_values)

        fps_in_mask_sample_point_idxs = self.toFPSPointIdxs(in_mask_sh_points, in_mask_sample_polar_idxs, self.sample_point_scale)

        in_mask_sample_points = in_mask_sh_points[fps_in_mask_sample_point_idxs]

        in_mask_sample_point_idxs = in_mask_sample_polar_idxs[fps_in_mask_sample_point_idxs]

        mask_boundary_sample_points = self.toForceSamplePoints(
            self.mask_boundary_phis, mask_boundary_thetas, self.mask_boundary_phi_idxs, self.mask_boundary_base_values)

        in_mask_x = in_mask_sample_points[:, 0]
        in_mask_y = in_mask_sample_points[:, 1]
        in_mask_z = in_mask_sample_points[:, 2]

        in_mask_sample_phis.grad = None
        in_mask_sample_theta_weights.grad = None
        in_mask_phi_grads_x = grad(in_mask_x, in_mask_sample_phis, torch.ones_like(in_mask_x), True)[0][fps_in_mask_sample_point_idxs].detach().clone()
        in_mask_theta_weight_grads_x = grad(in_mask_x, in_mask_sample_theta_weights, torch.ones_like(in_mask_x), True)[0][fps_in_mask_sample_point_idxs].detach().clone()

        in_mask_sample_phis.grad = None
        in_mask_sample_theta_weights.grad = None
        in_mask_phi_grads_y = grad(in_mask_y, in_mask_sample_phis, torch.ones_like(in_mask_y), True)[0][fps_in_mask_sample_point_idxs].detach().clone()
        in_mask_theta_weight_grads_y = grad(in_mask_y, in_mask_sample_theta_weights, torch.ones_like(in_mask_y), True)[0][fps_in_mask_sample_point_idxs].detach().clone()

        in_mask_sample_phis.grad = None
        in_mask_sample_theta_weights.grad = None
        in_mask_phi_grads_z = grad(in_mask_z, in_mask_sample_phis, torch.ones_like(in_mask_z), True)[0][fps_in_mask_sample_point_idxs].detach().clone()
        in_mask_theta_weight_grads_z = grad(in_mask_z, in_mask_sample_theta_weights, torch.ones_like(in_mask_z), True)[0][fps_in_mask_sample_point_idxs].detach().clone()

        mask_boundary_x = mask_boundary_sample_points[:, 0]
        mask_boundary_y = mask_boundary_sample_points[:, 1]
        mask_boundary_z = mask_boundary_sample_points[:, 2]

        self.mask_boundary_phis.grad = None
        mask_boundary_thetas.grad = None
        mask_boundary_phi_grads_x = grad(mask_boundary_x, self.mask_boundary_phis, torch.ones_like(mask_boundary_x), True)[0].detach().clone()
        mask_boundary_theta_grads_x = grad(mask_boundary_x, mask_boundary_thetas, torch.ones_like(mask_boundary_x), True)[0].detach().clone()

        self.mask_boundary_phis.grad = None
        mask_boundary_thetas.grad = None
        mask_boundary_phi_grads_y = grad(mask_boundary_y, self.mask_boundary_phis, torch.ones_like(mask_boundary_y), True)[0].detach().clone()
        mask_boundary_theta_grads_y = grad(mask_boundary_y, mask_boundary_thetas, torch.ones_like(mask_boundary_y), True)[0].detach().clone()

        self.mask_boundary_phis.grad = None
        mask_boundary_thetas.grad = None
        mask_boundary_phi_grads_z = grad(mask_boundary_z, self.mask_boundary_phis, torch.ones_like(mask_boundary_z), True)[0].detach().clone()
        mask_boundary_theta_grads_z = grad(mask_boundary_z, mask_boundary_thetas, torch.ones_like(mask_boundary_z), True)[0].detach().clone()

        self.mask_boundary_phis.requires_grad_(False)

        in_mask_nx = in_mask_phi_grads_y * in_mask_theta_weight_grads_z - in_mask_phi_grads_z * in_mask_theta_weight_grads_y
        in_mask_ny = in_mask_phi_grads_z * in_mask_theta_weight_grads_x - in_mask_phi_grads_x * in_mask_theta_weight_grads_z
        in_mask_nz = in_mask_phi_grads_x * in_mask_theta_weight_grads_y - in_mask_phi_grads_y * in_mask_theta_weight_grads_x

        mask_boundary_nx = mask_boundary_phi_grads_y * mask_boundary_theta_grads_z - mask_boundary_phi_grads_z * mask_boundary_theta_grads_y
        mask_boundary_ny = mask_boundary_phi_grads_z * mask_boundary_theta_grads_x - mask_boundary_phi_grads_x * mask_boundary_theta_grads_z
        mask_boundary_nz = mask_boundary_phi_grads_x * mask_boundary_theta_grads_y - mask_boundary_phi_grads_y * mask_boundary_theta_grads_x

        in_mask_normals = torch.vstack([in_mask_nx, in_mask_ny, in_mask_nz]).transpose(1, 0)
        mask_boundary_normals = torch.vstack([mask_boundary_nx, mask_boundary_ny, mask_boundary_nz]).transpose(1, 0)

        in_mask_norms = torch.norm(in_mask_normals, dim=1)
        mask_boundary_norms = torch.norm(mask_boundary_normals, dim=1)

        valid_in_mask_idxs = torch.where(in_mask_norms > 0)[0]
        valid_mask_boundary_idxs = torch.where(mask_boundary_norms > 0)[0]

        valid_in_mask_normals = torch.zeros_like(in_mask_sample_points)
        valid_in_mask_normals[valid_in_mask_idxs] = in_mask_normals[valid_in_mask_idxs] / in_mask_norms[valid_in_mask_idxs].reshape(-1, 1)
        valid_mask_boundary_normals = torch.zeros_like(mask_boundary_sample_points)
        valid_mask_boundary_normals[valid_mask_boundary_idxs] = mask_boundary_normals[valid_mask_boundary_idxs] / mask_boundary_norms[valid_mask_boundary_idxs].reshape(-1, 1)

        if refine_normals:
            normal_tags = toNormalTags(self.anchor_num, in_mask_sample_points, in_mask_sample_point_idxs, valid_in_mask_normals, fps_sample_scale, 'pgr')

            for i in range(self.anchor_num):
                anchor_boundary_normal_idxs = self.mask_boundary_phi_idxs == i
                anchor_inner_normal_idxs = in_mask_sample_point_idxs == i

                valid_mask_boundary_normals[anchor_boundary_normal_idxs] *= normal_tags[i]
                valid_in_mask_normals[anchor_inner_normal_idxs] *= normal_tags[i]

            if False:
                pcd = getPointCloud(toNumpy(in_mask_sample_points), toNumpy(valid_in_mask_normals))
                o3d.visualization.draw_geometries([pcd], point_show_normal=True)

        return (
            mask_boundary_sample_points,
            in_mask_sample_points,
            in_mask_sample_point_idxs,
            valid_mask_boundary_normals,
            valid_in_mask_normals,
        )

    def toSamplePoints(self, with_normals: bool = False, refine_normals: bool = False, fps_sample_scale: float = -1) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if with_normals:
            return self.toSamplePointsWithNormals(refine_normals, fps_sample_scale)

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
            self.rotate_vectors,
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
