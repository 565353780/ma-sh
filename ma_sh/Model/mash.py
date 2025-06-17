import os
import math
import torch
import numpy as np
import open3d as o3d
from tqdm import trange
from copy import deepcopy
from typing import Union, Optional

import mash_cpp

from ma_sh.Config.constant import W0, MAX_MASK_DEGREE, MAX_SH_DEGREE
from ma_sh.Method.data import toNumpy
from ma_sh.Method.check import checkShape
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Method.render import renderGeometries
from ma_sh.Method.rotate import compute_rotation_matrix_from_ortho6d
from ma_sh.Method.path import createFileFolder, removeFile, renameFile


class Mash(object):
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

        self.mask_params = torch.zeros(
            [anchor_num, 2 * mask_degree_max + 1],
            dtype=self.dtype,
            device=self.device,
        )
        self.sh_params = torch.zeros(
            [anchor_num, (sh_degree_max + 1) ** 2],
            dtype=self.dtype,
            device=self.device,
        )
        self.ortho_poses = torch.zeros(
            [anchor_num, 6], dtype=self.dtype, device=self.device
        )
        self.positions = torch.zeros(
            [anchor_num, 3], dtype=self.dtype, device=self.device
        )

        self.updatePreLoadDatas()

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

    @classmethod
    def fromMash(
        cls,
        target_mash,
        anchor_num: Union[int, None] = None,
        mask_degree_max: Union[int, None] = None,
        sh_degree_max: Union[int, None] = None,
        sample_phi_num: Union[int, None] = None,
        sample_theta_num: Union[int, None] = None,
        dtype=None,
        device: Union[str, None] = None,
    ):
        mash = cls(
            anchor_num if anchor_num is not None else target_mash.anchor_num,
            mask_degree_max
            if mask_degree_max is not None
            else target_mash.mask_degree_max,
            sh_degree_max if sh_degree_max is not None else target_mash.sh_degree_max,
            sample_phi_num
            if sample_phi_num is not None
            else target_mash.sample_phi_num,
            sample_theta_num
            if sample_theta_num is not None
            else target_mash.sample_theta_num,
            dtype if dtype is not None else target_mash.dtype,
            device if device is not None else target_mash.device,
        )
        return mash

    @classmethod
    def fromParamsDict(
        cls,
        params_dict: dict,
        sample_phi_num: int = 10,
        sample_theta_num: int = 10,
        dtype=torch.float32,
        device: str = "cpu",
    ):
        mask_params = params_dict["mask_params"]
        sh_params = params_dict["sh_params"]

        anchor_num = mask_params.shape[0]
        mask_degree_max = int((mask_params.shape[1] - 1) / 2)
        sh_degree_max = int(math.sqrt(sh_params.shape[1] - 1))

        mash = cls(
            anchor_num,
            mask_degree_max,
            sh_degree_max,
            sample_phi_num,
            sample_theta_num,
            dtype,
            device,
        )

        mash.loadParamsDict(params_dict)

        return mash

    @classmethod
    def fromParamsFile(
        cls,
        params_file_path: str,
        sample_phi_num: int = 10,
        sample_theta_num: int = 10,
        dtype=torch.float32,
        device: str = "cpu",
    ):
        params_dict = np.load(params_file_path, allow_pickle=True).item()

        return cls.fromParamsDict(
            params_dict,
            sample_phi_num,
            sample_theta_num,
            dtype,
            device,
        )

    def clone(self):
        return deepcopy(self)

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

    def editGrads(self, edit_fn, anchor_mask: Union[torch.Tensor, None] = None) -> bool:
        if anchor_mask is None:
            if self.mask_params.grad is not None:
                self.mask_params.grad = edit_fn(self.mask_params.grad)
            if self.sh_params.grad is not None:
                self.sh_params.grad = edit_fn(self.sh_params.grad)
            if self.ortho_poses.grad is not None:
                self.ortho_poses.grad = edit_fn(self.ortho_poses.grad)
            if self.positions.grad is not None:
                self.positions.grad = edit_fn(self.positions.grad)
            return True

        if self.mask_params[anchor_mask].grad is not None:
            self.mask_params[anchor_mask].grad = edit_fn(
                self.mask_params[anchor_mask].grad
            )
        if self.sh_params[anchor_mask].grad is not None:
            self.sh_params[anchor_mask].grad = edit_fn(self.sh_params[anchor_mask].grad)
        if self.ortho_poses[anchor_mask].grad is not None:
            self.ortho_poses[anchor_mask].grad = edit_fn(
                self.ortho_poses[anchor_mask].grad
            )
        if self.positions[anchor_mask].grad is not None:
            self.positions[anchor_mask].grad = edit_fn(self.positions[anchor_mask].grad)
        return True

    def clearGrads(self, anchor_mask: Union[torch.Tensor, None] = None) -> bool:
        def edit_fn(grad: torch.Tensor) -> None:
            return None

        return self.editGrads(edit_fn, anchor_mask)

    def clearNanGrads(self, anchor_mask: Union[torch.Tensor, None] = None) -> bool:
        def edit_fn(grad: torch.Tensor) -> torch.Tensor:
            nan_mask = torch.isnan(grad)
            grad[nan_mask] = 0
            return grad

        return self.editGrads(edit_fn, anchor_mask)

    def loadParams(
        self,
        mask_params: Union[torch.Tensor, np.ndarray, None] = None,
        sh_params: Union[torch.Tensor, np.ndarray, None] = None,
        ortho_poses: Union[torch.Tensor, np.ndarray, None] = None,
        positions: Union[torch.Tensor, np.ndarray, None] = None,
        face_forward_vectors: Union[torch.Tensor, np.ndarray, None] = None,
    ) -> bool:
        if ortho_poses is not None and face_forward_vectors is not None:
            print("[ERROR][Mash::loadParams]")
            print("\t ortho poses and face forward vectors are all None!")
            print("\t please make at least one of them be None!")
            return False

        if mask_params is not None:
            if not checkShape(mask_params.shape, self.mask_params.shape):
                print("[ERROR][Mash::loadParams]")
                print("\t checkShape failed for mask params!")
                return False

            if isinstance(mask_params, np.ndarray):
                mask_params = torch.from_numpy(mask_params)

            self.mask_params.data = (
                mask_params.detach().clone().type(self.dtype).to(self.device)
            )

        if sh_params is not None:
            if not checkShape(sh_params.shape, self.sh_params.shape):
                print("[ERROR][Mash::loadParams]")
                print("\t checkShape failed for sh params!")
                return False

            if isinstance(sh_params, np.ndarray):
                sh_params = torch.from_numpy(sh_params)

            self.sh_params.data = (
                sh_params.detach().clone().type(self.dtype).to(self.device)
            )

        if ortho_poses is not None:
            if not checkShape(ortho_poses.shape, self.ortho_poses.shape):
                print("[ERROR][Mash::loadParams]")
                print("\t checkShape failed for ortho poses!")
                return False

            if isinstance(ortho_poses, np.ndarray):
                ortho_poses = torch.from_numpy(ortho_poses)

            self.ortho_poses.data = (
                ortho_poses.detach().clone().type(self.dtype).to(self.device)
            )

        if positions is not None:
            if not checkShape(positions.shape, self.positions.shape):
                print("[ERROR][Mash::loadParams]")
                print("\t checkShape failed for positions!")
                return False

            if isinstance(positions, np.ndarray):
                positions = torch.from_numpy(positions)

            self.positions.data = (
                positions.detach().clone().type(self.dtype).to(self.device)
            )

        if face_forward_vectors is not None:
            if not checkShape(face_forward_vectors.shape, self.positions.shape):
                print("[ERROR][Mash::loadParams]")
                print("\t checkShape failed for face forward vectors!")
                return False

            if isinstance(face_forward_vectors, np.ndarray):
                face_forward_vectors = torch.from_numpy(face_forward_vectors)

            rotate_vectors = mash_cpp.toRotateVectorsByFaceForwardVectors(
                face_forward_vectors
            )

            rotate_matrixs = mash_cpp.toRotateMatrixs(rotate_vectors)

            ortho_poses = rotate_matrixs[:, :2, :].view(-1, 6)

            self.ortho_poses.data = (
                ortho_poses.detach().clone().type(self.dtype).to(self.device)
            )
        return True

    def loadParamsDict(self, params_dict: dict) -> bool:
        if "mask_params" not in params_dict.keys():
            print("[ERROR][Mash::loadParamsDict]")
            print("\t mask_params not in params dict!")
            return False

        if "sh_params" not in params_dict.keys():
            print("[ERROR][Mash::loadParamsDict]")
            print("\t sh_params not in params dict!")
            return False

        if "ortho_poses" not in params_dict.keys():
            print("[ERROR][Mash::loadParamsDict]")
            print("\t ortho_poses not in params dict!")
            return False

        if "positions" not in params_dict.keys():
            print("[ERROR][Mash::loadParamsDict]")
            print("\t positions not in params dict!")
            return False

        mask_params = params_dict["mask_params"]
        sh_params = params_dict["sh_params"]
        ortho_poses = params_dict["ortho_poses"]
        positions = params_dict["positions"]

        if not self.loadParams(mask_params, sh_params, ortho_poses, positions):
            print("[ERROR][Mash::loadParamsDict]")
            print("\t loadParams failed!")
            return False

        return True

    def mergeMash(self, mash) -> bool:
        if self.mask_degree_max != mash.mask_degree_max:
            print("[ERROR][Mash::mergeMash]")
            print("\t mask degree max not matched!")
            print("\t ", self.mask_degree_max, "!=", mash.mask_degree_max)
            return False

        if self.sh_degree_max != mash.sh_degree_max:
            print("[ERROR][Mash::mergeMash]")
            print("\t sh degree max not matched!")
            print("\t ", self.sh_degree_max, "!=", mash.sh_degree_max)
            return False

        self.setGradState(False)

        self.anchor_num += mash.anchor_num
        self.mask_params = torch.vstack(
            [self.mask_params, mash.mask_params.detach().clone()]
        )
        self.sh_params = torch.vstack([self.sh_params, mash.sh_params.detach().clone()])
        self.positions = torch.vstack([self.positions, mash.positions.detach().clone()])
        self.rotate_vectors = torch.vstack(
            [self.rotate_vectors, mash.rotate_vectors.detach().clone()]
        )

        self.updatePreLoadDatas()

        return True

    def updateAnchorNum(self, anchor_num: int) -> bool:
        if anchor_num == self.anchor_num:
            return True

        if anchor_num < 1:
            print("[ERROR][Mash::updateAnchorNum]")
            print("\t anchor num < 1!")
            print("\t anchor_num:", anchor_num)
            return False

        copy_dim = min(anchor_num, self.anchor_num)

        self.anchor_num = anchor_num

        new_mask_params, new_sh_params, new_rotate_vectors, new_positions = toParams(
            self.anchor_num,
            self.mask_degree_max,
            self.sh_degree_max,
            self.dtype,
            self.device,
        )

        new_mask_params[:copy_dim, :] = self.mask_params[:copy_dim, :]
        new_sh_params[:copy_dim, :] = self.sh_params[:copy_dim, :]
        new_positions[:copy_dim, :] = self.positions[:copy_dim, :]
        new_rotate_vectors[:copy_dim, :] = self.rotate_vectors[:copy_dim, :]

        self.mask_params = new_mask_params
        self.sh_params = new_sh_params
        self.positions = new_positions
        self.rotate_vectors = new_rotate_vectors

        self.updatePreLoadDatas()

        return True

    def updateMaskDegree(self, mask_degree_max: int) -> bool:
        if mask_degree_max == self.mask_degree_max:
            return True

        if mask_degree_max < 0 or mask_degree_max > MAX_MASK_DEGREE:
            print("[ERROR][Mash::updateMaskDegree]")
            print("\t mask degree max out of range!")
            print("\t mask_degree_max:", mask_degree_max)
            return False

        self.mask_degree_max = mask_degree_max

        new_dim = 2 * self.mask_degree_max + 1

        new_mask_params = torch.zeros([self.anchor_num, new_dim], dtype=self.dtype).to(
            self.device
        )

        copy_dim = min(new_dim, self.mask_params.shape[1])

        new_mask_params[:, :copy_dim] = self.mask_params.detach().clone()

        self.mask_params = new_mask_params

        self.updatePreLoadDatas()
        return True

    def updateSHDegree(self, sh_degree_max: int) -> bool:
        if sh_degree_max == self.sh_degree_max:
            return True

        if sh_degree_max < 0 or sh_degree_max > MAX_SH_DEGREE:
            print("[ERROR][Mash::updateSHDegree]")
            print("\t sh degree max out of range!")
            print("\t sh_degree_max:", sh_degree_max)
            return False

        self.sh_degree_max = sh_degree_max

        new_dim = (self.sh_degree_max + 1) ** 2

        new_sh_params = torch.zeros([self.anchor_num, new_dim], dtype=self.dtype).to(
            self.device
        )

        copy_dim = min(new_dim, self.sh_params.shape[1])

        new_sh_params[:, :copy_dim] = self.sh_params.detach().clone()

        self.sh_params = new_sh_params
        return True

    def toCenter(self) -> torch.Tensor:
        min_bound = torch.min(self.positions.data, dim=0)[0]
        max_bound = torch.max(self.positions.data, dim=0)[0]
        center = (min_bound + max_bound) / 2.0
        return center

    def translate(self, move_position: Union[torch.Tensor, np.ndarray, list]) -> bool:
        if isinstance(move_position, list):
            move_position = np.array(move_position)

        if isinstance(move_position, np.ndarray):
            move_position = torch.from_numpy(move_position)

        move_position = move_position.type(self.positions.dtype).to(
            self.positions.device
        )

        grad_state = self.positions.requires_grad

        self.setGradState(False)

        self.positions.data += move_position.unsqueeze(0)

        self.setGradState(grad_state)
        return True

    def scale(self, scale: float, keep_position: bool = True) -> bool:
        grad_state = self.positions.requires_grad

        self.setGradState(False)

        move_position = None
        if keep_position:
            move_position = self.toCenter()
            self.translate(-1.0 * move_position)

        self.positions.data *= scale
        self.sh_params.data *= scale

        if move_position is not None:
            self.translate(move_position)

        self.setGradState(grad_state)
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
        sample_move_vectors = self.toInvSampleMoveVectors()

        rotate_mats = compute_rotation_matrix_from_ortho6d(self.ortho_poses)

        rotated_sample_move_vectors = torch.einsum(
            "b...i,bij->b...j", sample_move_vectors, rotate_mats
        )

        positions_expanded = self.positions.view(
            self.anchor_num, *((1,) * (sample_move_vectors.dim() - 2)), 3
        )
        sample_points = positions_expanded + rotated_sample_move_vectors

        return sample_points

    def toSamplePcd(self) -> o3d.geometry.PointCloud:
        sample_points = self.toSamplePoints().reshape(-1, 3)

        sample_points_array = toNumpy(sample_points)

        sample_pcd = getPointCloud(sample_points_array)
        return sample_pcd

    def renderSamplePoints(self) -> bool:
        sample_pts = toNumpy(self.toSamplePoints()).reshape(-1, 3)

        pcd = getPointCloud(sample_pts)
        renderGeometries(pcd)
        return True

    def toParamsDict(self) -> dict:
        params_dict = {
            "mask_params": toNumpy(self.mask_params),
            "sh_params": toNumpy(self.sh_params),
            "ortho_poses": toNumpy(self.ortho_poses),
            "positions": toNumpy(self.positions),
        }
        return params_dict

    def saveParamsFile(
        self,
        save_params_file_path: str,
        overwrite: bool = False,
    ) -> bool:
        if os.path.exists(save_params_file_path):
            if not overwrite:
                return True

            removeFile(save_params_file_path)

        params_dict = self.toParamsDict()

        createFileFolder(save_params_file_path)

        tmp_save_params_file_path = save_params_file_path[:-4] + "_tmp.npy"
        removeFile(tmp_save_params_file_path)

        np.save(tmp_save_params_file_path, params_dict)
        renameFile(tmp_save_params_file_path, save_params_file_path)
        return True

    @torch.no_grad()
    def saveAsPcdFile(
        self,
        save_pcd_file_path: str,
        overwrite: bool = False,
        print_progress: bool = False,
    ) -> bool:
        if os.path.exists(save_pcd_file_path):
            if not overwrite:
                return True

            removeFile(save_pcd_file_path)

        createFileFolder(save_pcd_file_path)

        pcd = self.toSamplePcd()

        if print_progress:
            print("[INFO][Mash::saveAsPcdFile]")
            print("\t start save as pcd file...")

        tmp_save_pcd_file_path = (
            save_pcd_file_path[:-4] + "_tmp" + save_pcd_file_path[-4:]
        )

        o3d.io.write_point_cloud(
            tmp_save_pcd_file_path, pcd, write_ascii=True, print_progress=print_progress
        )

        renameFile(tmp_save_pcd_file_path, save_pcd_file_path)
        return True

    @torch.no_grad()
    def saveAsAnchorPcdFiles(
        self,
        save_pcd_folder_path: str,
        overwrite: bool = False,
    ) -> bool:
        os.makedirs(save_pcd_folder_path, exist_ok=True)

        sample_points = self.toSamplePoints().reshape(self.anchor_num, -1, 3)

        print("[INFO][Mash::saveAsAnchorPcdFiles]")
        print("\t start save as anchor pcd files...")
        for i in trange(self.anchor_num):
            save_pcd_file_path = save_pcd_folder_path + str(i) + "_pcd.ply"
            if os.path.exists(save_pcd_file_path):
                if not overwrite:
                    continue

                removeFile(save_pcd_file_path)

            pcd = getPointCloud(sample_points[i])

            tmp_save_pcd_file_path = (
                save_pcd_file_path[:-4] + "_tmp" + save_pcd_file_path[-4:]
            )

            o3d.io.write_point_cloud(tmp_save_pcd_file_path, pcd, write_ascii=True)

            if not os.path.exists(tmp_save_pcd_file_path):
                print("[ERROR][Mash::saveAsAnchorPcdFiles]")
                print("\t write_point_cloud failed!")
                print("\t save_pcd_file_path:", save_pcd_file_path)
                continue

            renameFile(tmp_save_pcd_file_path, save_pcd_file_path)

        return True
