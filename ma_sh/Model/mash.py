import os
import torch
import numpy as np
from math import sqrt
from typing import Union

import mash_cpp

from ma_sh.Config.degree import MAX_MASK_DEGREE, MAX_SH_DEGREE
from ma_sh.Method.check import checkShape
from ma_sh.Method.Mash.mash import toParams, toPreLoadDatas
from ma_sh.Method.render import renderPoints
from ma_sh.Method.path import createFileFolder, removeFile, renameFile


class Mash(object):
    def __init__(
        self,
        anchor_num: int,
        mask_degree_max: int,
        sh_degree_max: int,
        mask_boundary_sample_num: int = 36,
        sample_polar_num: int = 2000,
        sample_point_scale: float = 0.8,
        use_inv: bool = True,
        idx_dtype=torch.int64,
        dtype=torch.float64,
        device: str = "cpu",
    ) -> None:
        # Super Params
        self.anchor_num = anchor_num
        self.mask_degree_max = mask_degree_max
        self.sh_degree_max = sh_degree_max
        self.mask_boundary_sample_num = mask_boundary_sample_num
        self.sample_polar_num = sample_polar_num
        self.sample_point_scale = sample_point_scale
        self.use_inv = use_inv
        self.idx_dtype = idx_dtype
        self.dtype = dtype
        self.device = device

        # Diff Params
        self.mask_params = torch.tensor([0.0], dtype=dtype).to(self.device)
        self.sh_params = torch.tensor([0.0], dtype=dtype).to(self.device)
        self.rotate_vectors = torch.tensor([0.0], dtype=dtype).to(self.device)
        self.positions = torch.tensor([0.0], dtype=dtype).to(self.device)

        # Pre Load Datas
        self.sample_phis = torch.tensor([0.0], dtype=dtype).to(self.device)
        self.sample_thetas = torch.tensor([0.0], dtype=dtype).to(self.device)
        self.mask_boundary_phis = torch.tensor([0.0], dtype=dtype).to(self.device)
        self.mask_boundary_phi_idxs = torch.tensor([0.0], dtype=dtype).to(self.device)
        self.mask_boundary_base_values = torch.tensor([0.0], dtype=dtype).to(
            self.device
        )
        self.sample_base_values = torch.tensor([0.0], dtype=dtype).to(self.device)
        self.sample_sh_directions = torch.tensor([0.0], dtype=dtype).to(self.device)

        self.reset()
        return

    @classmethod
    def fromParamsDict(
        cls,
        params_dict: dict,
        mask_boundary_sample_num: int = 36,
        sample_polar_num: int = 2000,
        sample_point_scale: float = 0.8,
        idx_dtype=torch.int64,
        dtype=torch.float64,
        device: str = "cuda:0",
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
        mask_boundary_sample_num: int = 36,
        sample_polar_num: int = 2000,
        sample_point_scale: float = 0.8,
        idx_dtype=torch.int64,
        dtype=torch.float64,
        device: str = "cuda:0",
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

    def reset(self) -> bool:
        self.initParams()
        self.updatePreLoadDatas()
        return True

    def setGradState(self, need_grad: bool) -> bool:
        self.mask_params.requires_grad_(need_grad)
        self.sh_params.requires_grad_(need_grad)
        self.rotate_vectors.requires_grad_(need_grad)
        self.positions.requires_grad_(need_grad)
        return True

    def initParams(self) -> bool:
        self.mask_params, self.sh_params, self.rotate_vectors, self.positions = (
            toParams(
                self.anchor_num,
                self.mask_degree_max,
                self.sh_degree_max,
                self.dtype,
                self.device,
            )
        )
        return True

    def updatePreLoadDatas(self) -> bool:
        (
            self.sample_phis,
            self.sample_thetas,
            self.mask_boundary_phis,
            self.mask_boundary_phi_idxs,
            self.mask_boundary_base_values,
            self.sample_base_values,
            self.sample_sh_directions,
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

    def loadParams(
        self,
        mask_params: Union[torch.Tensor, np.ndarray, None] = None,
        sh_params: Union[torch.Tensor, np.ndarray, None] = None,
        rotate_vectors: Union[torch.Tensor, np.ndarray, None] = None,
        positions: Union[torch.Tensor, np.ndarray, None] = None,
        use_inv: Union[bool, None] = None,
        face_forward_vectors: Union[torch.Tensor, np.ndarray, None] = None,
    ) -> bool:
        if rotate_vectors is not None and face_forward_vectors is not None:
            print("[ERROR][Mash::loadParams]")
            print("\t rotate vectors and face forward vectors are all None!")
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

        if rotate_vectors is not None:
            if not checkShape(rotate_vectors.shape, self.rotate_vectors.shape):
                print("[ERROR][Mash::loadParams]")
                print("\t checkShape failed for rotate vectors!")
                return False

            if isinstance(rotate_vectors, np.ndarray):
                rotate_vectors = torch.from_numpy(rotate_vectors)

            self.rotate_vectors.data = (
                rotate_vectors.detach().clone().type(self.dtype).to(self.device)
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

        if use_inv is not None:
            self.use_inv = use_inv

        if face_forward_vectors is not None:
            if not checkShape(face_forward_vectors.shape, self.rotate_vectors.shape):
                print("[ERROR][Mash::loadParams]")
                print("\t checkShape failed for face forward vectors!")
                return False

            if isinstance(face_forward_vectors, np.ndarray):
                face_forward_vectors = torch.from_numpy(face_forward_vectors)

            rotate_vectors = mash_cpp.toRotateVectorsByFaceForwardVectors(
                face_forward_vectors
            )

            self.rotate_vectors.data = (
                rotate_vectors.detach().clone().type(self.dtype).to(self.device)
            )

        return True

    def loadParamsDict(self, params_dict: dict) -> bool:
        mask_params = params_dict["mask_params"]
        sh_params = params_dict["sh_params"]
        rotate_vectors = params_dict["rotate_vectors"]
        positions = params_dict["positions"]
        use_inv = params_dict["use_inv"]

        self.loadParams(mask_params, sh_params, rotate_vectors, positions, use_inv)

        return True

    def loadParamsFile(self, params_file_path: str) -> bool:
        if not os.path.exists(params_file_path):
            print("[ERROR][Mash::loadParamsFile]")
            print("\t params dict file not exist!")
            print("\t params_file_path:", params_file_path)
            return False

        params_dict = np.load(params_file_path, allow_pickle=True).item()

        if not self.loadParamsDict(params_dict):
            print("[ERROR][Mash::loadParamsFile]")
            print("\t loadParamsDict failed!")
            return False

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

        new_mask_params = torch.zeros(
            [self.mask_params.shape[0], new_dim], dtype=self.dtype
        ).to(self.device)

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

        new_sh_params = torch.zeros(
            [self.sh_params.shape[0], new_dim], dtype=self.dtype
        ).to(self.device)

        copy_dim = min(new_dim, self.sh_params.shape[1])

        new_sh_params[:, :copy_dim] = self.sh_params.detach().clone()

        self.sh_params = new_sh_params
        return True

    def toSamplePoints(self) -> torch.Tensor:
        sample_points = mash_cpp.toMashSamplePoints(
            self.anchor_num,
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
            self.sample_sh_directions,
            self.sample_point_scale,
            self.use_inv,
        )

        return sample_points

    def renderSamplePoints(self) -> bool:
        sample_points = self.toSamplePoints().detach().clone().cpu().numpy()
        print(sample_points.shape)

        renderPoints(sample_points)
        return True

    def toParamsDict(self) -> dict:
        params_dict = {
            "mask_params": self.mask_params.detach().clone().cpu().numpy(),
            "sh_params": self.sh_params.detach().clone().cpu().numpy(),
            "rotate_vectors": self.rotate_vectors.detach().clone().cpu().numpy(),
            "positions": self.positions.detach().clone().cpu().numpy(),
            "use_inv": self.use_inv,
        }
        return params_dict

    def saveParamsFile(
        self, save_params_file_path: str, overwrite: bool = False
    ) -> bool:
        if os.path.exists(save_params_file_path):
            if overwrite:
                removeFile(save_params_file_path)
            else:
                print("[WARN][Mash::saveParamsFile]")
                print("\t save params dict file already exist!")
                print("\t save_params_file_path:", save_params_file_path)
                return False

        params_dict = self.toParamsDict()

        createFileFolder(save_params_file_path)

        tmp_save_params_file_path = save_params_file_path[:-4] + "_tmp.npy"
        removeFile(tmp_save_params_file_path)

        np.save(tmp_save_params_file_path, params_dict)
        renameFile(tmp_save_params_file_path, save_params_file_path)
        return True
