import os
import torch
import numpy as np
import open3d as o3d
from math import sqrt
from copy import deepcopy
from typing import Union
from abc import ABC, abstractmethod

import mash_cpp

from ma_sh.Config.constant import MAX_MASK_DEGREE, MAX_SH_DEGREE
from ma_sh.Method.data import toNumpy
from ma_sh.Method.check import checkShape
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Method.render import renderGeometries
from ma_sh.Method.path import createFileFolder, removeFile, renameFile
from ma_sh.Method.rotate import (
    toOrthoPosesFromRotateVectors,
    toRotateVectorsFromOrthoPoses,
)


class BaseMash(ABC):
    def __init__(
        self,
        anchor_num: int = 400,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        use_inv: bool = True,
        dtype=torch.float64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        # Super Params
        self.anchor_num = anchor_num
        self.mask_degree_max = mask_degree_max
        self.sh_degree_max = sh_degree_max
        self.use_inv = use_inv
        self.dtype = dtype
        self.device = device

        # Diff Params
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

        # Pre Load Datas
        self.updatePreLoadDatas()
        return

    @classmethod
    def fromMash(
        cls,
        target_mash,
        anchor_num: Union[int, None] = None,
        mask_degree_max: Union[int, None] = None,
        sh_degree_max: Union[int, None] = None,
        use_inv: Union[bool, None] = None,
        dtype=None,
        device: Union[str, None] = None,
    ):
        mash = cls(
            anchor_num if anchor_num is not None else target_mash.anchor_num,
            mask_degree_max
            if mask_degree_max is not None
            else target_mash.mask_degree_max,
            sh_degree_max if sh_degree_max is not None else target_mash.sh_degree_max,
            use_inv if use_inv is not None else target_mash.use_inv,
            dtype if dtype is not None else target_mash.dtype,
            device if device is not None else target_mash.device,
        )
        return mash

    @classmethod
    def fromParamsDict(
        cls,
        params_dict: dict,
        dtype=torch.float64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
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
            use_inv,
            dtype,
            device,
        )

        mash.loadParamsDict(params_dict)

        return mash

    @classmethod
    def fromParamsFile(
        cls,
        params_file_path: str,
        dtype=torch.float64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        params_dict = np.load(params_file_path, allow_pickle=True).item()

        return cls.fromParamsDict(
            params_dict,
            dtype,
            device,
        )

    def clone(self):
        return deepcopy(self)

    def setGradState(
        self, need_grad: bool, anchor_mask: Union[torch.Tensor, None] = None
    ) -> bool:
        if anchor_mask is None:
            self.mask_params.requires_grad_(need_grad)
            self.sh_params.requires_grad_(need_grad)
            self.ortho_poses.requires_grad_(need_grad)
            self.positions.requires_grad_(need_grad)
            return True

        self.mask_params[anchor_mask].requires_grad_(need_grad)
        self.sh_params[anchor_mask].requires_grad_(need_grad)
        self.ortho_poses[anchor_mask].requires_grad_(need_grad)
        self.positions[anchor_mask].requires_grad_(need_grad)
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

    def randomInit(self) -> bool:
        self.mask_params = torch.randn_like(self.mask_params)
        self.sh_params = torch.randn_like(self.sh_params)
        self.ortho_poses = torch.randn_like(self.ortho_poses)
        self.positions = torch.randn_like(self.positions)
        return True

    @abstractmethod
    def updatePreLoadDatas(self) -> bool:
        pass

    def loadParams(
        self,
        mask_params: Union[torch.Tensor, np.ndarray, None] = None,
        sh_params: Union[torch.Tensor, np.ndarray, None] = None,
        ortho_poses: Union[torch.Tensor, np.ndarray, None] = None,
        positions: Union[torch.Tensor, np.ndarray, None] = None,
        use_inv: Union[bool, None] = None,
        face_forward_vectors: Union[torch.Tensor, np.ndarray, None] = None,
    ) -> bool:
        if ortho_poses is not None and face_forward_vectors is not None:
            print("[ERROR][BaseMash::loadParams]")
            print("\t ortho poses and face forward vectors are all None!")
            print("\t please make at least one of them be None!")
            return False

        if mask_params is not None:
            if not checkShape(mask_params.shape, self.mask_params.shape):
                print("[ERROR][BaseMash::loadParams]")
                print("\t checkShape failed for mask params!")
                return False

            if isinstance(mask_params, np.ndarray):
                mask_params = torch.from_numpy(mask_params)

            self.mask_params.data = (
                mask_params.detach().clone().type(self.dtype).to(self.device)
            )

        if sh_params is not None:
            if not checkShape(sh_params.shape, self.sh_params.shape):
                print("[ERROR][BaseMash::loadParams]")
                print("\t checkShape failed for sh params!")
                return False

            if isinstance(sh_params, np.ndarray):
                sh_params = torch.from_numpy(sh_params)

            self.sh_params.data = (
                sh_params.detach().clone().type(self.dtype).to(self.device)
            )

        if ortho_poses is not None:
            if not checkShape(ortho_poses.shape, self.ortho_poses.shape):
                print("[ERROR][BaseMash::loadParams]")
                print("\t checkShape failed for ortho poses!")
                return False

            if isinstance(ortho_poses, np.ndarray):
                ortho_poses = torch.from_numpy(ortho_poses)

            self.ortho_poses.data = (
                ortho_poses.detach().clone().type(self.dtype).to(self.device)
            )

        if positions is not None:
            if not checkShape(positions.shape, self.positions.shape):
                print("[ERROR][BaseMash::loadParams]")
                print("\t checkShape failed for positions!")
                return False

            if isinstance(positions, np.ndarray):
                positions = torch.from_numpy(positions)

            self.positions.data = (
                positions.detach().clone().type(self.dtype).to(self.device)
            )

        if use_inv is not None:
            self.use_inv = use_inv

        if face_forward_vectors is not None:
            if not checkShape(face_forward_vectors.shape, self.positions.shape):
                print("[ERROR][BaseMash::loadParams]")
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
            print("[ERROR][BaseMash::loadParamsDict]")
            print("\t mask_params not in params dict!")
            return False

        if "sh_params" not in params_dict.keys():
            print("[ERROR][BaseMash::loadParamsDict]")
            print("\t sh_params not in params dict!")
            return False

        # FIXME: tmp use the old MASH data for archive dataset
        if "rotate_vectors" in params_dict.keys():
            rotate_vectors = -1.0 * torch.from_numpy(params_dict["rotate_vectors"])
            params_dict["ortho_poses"] = toOrthoPosesFromRotateVectors(rotate_vectors)

        if "ortho_poses" not in params_dict.keys():
            print("[ERROR][BaseMash::loadParamsDict]")
            print("\t ortho_poses not in params dict!")
            return False

        if "positions" not in params_dict.keys():
            print("[ERROR][BaseMash::loadParamsDict]")
            print("\t positions not in params dict!")
            return False

        if "use_inv" not in params_dict.keys():
            print("[ERROR][BaseMash::loadParamsDict]")
            print("\t use_inv not in params dict!")
            return False

        mask_params = params_dict["mask_params"]
        sh_params = params_dict["sh_params"]
        ortho_poses = params_dict["ortho_poses"]
        positions = params_dict["positions"]
        use_inv = params_dict["use_inv"]

        if not self.loadParams(mask_params, sh_params, ortho_poses, positions, use_inv):
            print("[ERROR][BaseMash::loadParamsDict]")
            print("\t loadParams failed!")
            return False

        return True

    def loadParamsFile(self, params_file_path: str) -> bool:
        if not os.path.exists(params_file_path):
            print("[ERROR][BaseMash::loadParamsFile]")
            print("\t params dict file not exist!")
            print("\t params_file_path:", params_file_path)
            return False

        params_dict = np.load(params_file_path, allow_pickle=True).item()

        if not self.loadParamsDict(params_dict):
            print("[ERROR][BaseMash::loadParamsFile]")
            print("\t loadParamsDict failed!")
            return False

        return True

    def mergeMash(self, mash) -> bool:
        if self.mask_degree_max != mash.mask_degree_max:
            print("[ERROR][BaseMash::mergeMash]")
            print("\t mask degree max not matched!")
            print("\t ", self.mask_degree_max, "!=", mash.mask_degree_max)
            return False

        if self.sh_degree_max != mash.sh_degree_max:
            print("[ERROR][BaseMash::mergeMash]")
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
        self.ortho_poses = torch.vstack(
            [self.ortho_poses, mash.ortho_poses.detach().clone()]
        )

        self.updatePreLoadDatas()

        return True

    def updateAnchorNum(self, anchor_num: int) -> bool:
        if anchor_num == self.anchor_num:
            return True

        if anchor_num < 1:
            print("[ERROR][BaseMash::updateAnchorNum]")
            print("\t anchor num < 1!")
            print("\t anchor_num:", anchor_num)
            return False

        copy_dim = min(anchor_num, self.anchor_num)

        self.anchor_num = anchor_num

        new_mask_params = torch.zeros(
            [self.anchor_num, 2 * self.mask_degree_max + 1],
            dtype=self.dtype,
            device=self.device,
        )
        new_sh_params = torch.zeros(
            [self.anchor_num, (self.sh_degree_max + 1) ** 2],
            dtype=self.dtype,
            device=self.device,
        )
        new_ortho_poses = torch.zeros(
            [self.anchor_num, 6], dtype=self.dtype, device=self.device
        )
        new_positions = torch.zeros(
            [self.anchor_num, 3], dtype=self.dtype, device=self.device
        )

        new_mask_params[:copy_dim, :] = self.mask_params[:copy_dim, :]
        new_sh_params[:copy_dim, :] = self.sh_params[:copy_dim, :]
        new_positions[:copy_dim, :] = self.positions[:copy_dim, :]
        new_ortho_poses[:copy_dim, :] = self.ortho_poses[:copy_dim, :]

        self.mask_params = new_mask_params
        self.sh_params = new_sh_params
        self.positions = new_positions
        self.ortho_poses = new_ortho_poses

        self.updatePreLoadDatas()

        return True

    def updateMaskDegree(self, mask_degree_max: int) -> bool:
        if mask_degree_max == self.mask_degree_max:
            return True

        if mask_degree_max < 0 or mask_degree_max > MAX_MASK_DEGREE:
            print("[ERROR][BaseMash::updateMaskDegree]")
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
            print("[ERROR][BaseMash::updateSHDegree]")
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

    def toFaceToPoints(self) -> torch.Tensor:
        rotate_vectors = toRotateVectorsFromOrthoPoses(self.ortho_poses)

        face_to_points = mash_cpp.toFaceToPoints(
            self.mask_degree_max,
            self.sh_degree_max,
            self.sh_params,
            rotate_vectors,
            self.positions,
            self.use_inv,
        )
        return face_to_points

    def toWeightedSamplePoints(
        self,
        sample_phis: torch.Tensor,
        sample_theta_weights: torch.Tensor,
        sample_idxs: torch.Tensor,
        sample_base_values: torch.Tensor = torch.Tensor(),
    ) -> torch.Tensor:
        rotate_vectors = toRotateVectorsFromOrthoPoses(self.ortho_poses)

        weighted_sample_points = mash_cpp.toWeightedSamplePoints(
            self.mask_degree_max,
            self.sh_degree_max,
            self.mask_params,
            self.sh_params,
            rotate_vectors,
            self.positions,
            sample_phis,
            sample_theta_weights,
            sample_idxs,
            self.use_inv,
            sample_base_values,
        )

        return weighted_sample_points

    def toForceSamplePoints(
        self,
        sample_phis: torch.Tensor,
        sample_thetas: torch.Tensor,
        sample_idxs: torch.Tensor,
        sample_base_values: torch.Tensor = torch.Tensor(),
    ) -> torch.Tensor:
        rotate_vectors = toRotateVectorsFromOrthoPoses(self.ortho_poses)

        sample_points = mash_cpp.toSamplePoints(
            self.mask_degree_max,
            self.sh_degree_max,
            self.sh_params,
            rotate_vectors,
            self.positions,
            sample_phis,
            sample_thetas,
            sample_idxs,
            self.use_inv,
            sample_base_values,
        )
        return sample_points

    def toFPSPointIdxs(
        self,
        sample_points: torch.Tensor,
        sample_idxs: torch.Tensor,
        sample_point_scale: float,
    ) -> torch.Tensor:
        fps_sample_point_idxs = mash_cpp.toFPSPointIdxs(
            sample_points, sample_idxs, sample_point_scale, self.anchor_num
        )
        return fps_sample_point_idxs

    @abstractmethod
    def toSamplePoints(
        self,
    ) -> torch.Tensor:
        pass

    def toSamplePcd(self) -> o3d.geometry.PointCloud:
        sample_points = self.toSamplePoints().reshape(-1, 3)

        sample_points_array = toNumpy(sample_points)

        sample_pcd = getPointCloud(sample_points_array)
        return sample_pcd

    def renderSamplePoints(self) -> bool:
        sample_pcd = self.toSamplePcd()
        renderGeometries(sample_pcd)
        return True

    def toParamsDict(self) -> dict:
        params_dict = {
            "mask_params": toNumpy(self.mask_params),
            "sh_params": toNumpy(self.sh_params),
            "ortho_poses": toNumpy(self.ortho_poses),
            "positions": toNumpy(self.positions),
            "use_inv": self.use_inv,
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
            print("[INFO][BaseMash::saveAsPcdFile]")
            print("\t start save as pcd file...")

        tmp_save_pcd_file_path = (
            save_pcd_file_path[:-4] + "_tmp" + save_pcd_file_path[-4:]
        )

        o3d.io.write_point_cloud(
            tmp_save_pcd_file_path, pcd, write_ascii=True, print_progress=print_progress
        )

        renameFile(tmp_save_pcd_file_path, save_pcd_file_path)
        return True
