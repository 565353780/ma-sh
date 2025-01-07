import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from math import sqrt
from copy import deepcopy
from typing import Union, Tuple
from abc import ABC, abstractmethod

import mash_cpp

from ma_sh.Config.degree import MAX_MASK_DEGREE, MAX_SH_DEGREE
from ma_sh.Method.data import toNumpy
from ma_sh.Method.check import checkShape
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Method.Mash.mash import toParams
from ma_sh.Method.render import renderGeometries
from ma_sh.Method.path import createFileFolder, removeFile, renameFile
from ma_sh.Method.rotate import (
    toRegularRotateVectors,
    toOrthoPosesFromRotateVectors,
    toRotateVectorsFromOrthoPoses
)


class BaseMash(ABC):
    def __init__(
        self,
        anchor_num: int = 400,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        use_inv: bool = True,
        idx_dtype=torch.int64,
        dtype=torch.float64,
        device: str = "cpu",
    ) -> None:
        # Super Params
        self.anchor_num = anchor_num
        self.mask_degree_max = mask_degree_max
        self.sh_degree_max = sh_degree_max
        self.use_inv = use_inv
        self.idx_dtype = idx_dtype
        self.dtype = dtype
        self.device = device

        # Diff Params
        self.mask_params = torch.tensor([0.0], dtype=self.dtype).to(self.device)
        self.sh_params = torch.tensor([0.0], dtype=self.dtype).to(self.device)
        self.rotate_vectors = torch.tensor([0.0], dtype=self.dtype).to(self.device)
        self.positions = torch.tensor([0.0], dtype=self.dtype).to(self.device)

        # Pre Load Datas
        self.sample_phis = torch.tensor([0.0], dtype=self.dtype).to(self.device)
        self.sample_base_values = torch.tensor([0.0], dtype=self.dtype).to(self.device)
        self.mask_boundary_phi_idxs = torch.tensor([0.0], dtype=self.dtype).to(self.device)

        self.reset()
        return

    @classmethod
    def fromMash(
        cls,
        target_mash,
        anchor_num: Union[int, None] = None,
        mask_degree_max: Union[int, None] = None,
        sh_degree_max: Union[int, None] = None,
        use_inv: Union[bool, None] = None,
        idx_dtype=None,
        dtype=None,
        device: Union[str, None] = None,
        ):

        mash = cls(
            anchor_num if anchor_num is not None else target_mash.anchor_num,
            mask_degree_max if mask_degree_max is not None else target_mash.mask_degree_max,
            sh_degree_max if sh_degree_max is not None else target_mash.sh_degree_max,
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
        idx_dtype=torch.int64,
        dtype=torch.float64,
        device: str = "cpu",
    ):
        params_dict = np.load(params_file_path, allow_pickle=True).item()

        return cls.fromParamsDict(
            params_dict,
            idx_dtype,
            dtype,
            device,
        )

    def reset(self) -> bool:
        self.initParams()
        self.updatePreLoadDatas()
        return True

    def clone(self):
        return deepcopy(self)

    def setGradState(self, need_grad: bool, anchor_mask: Union[torch.Tensor, None] = None) -> bool:
        if anchor_mask is None:
            self.mask_params.requires_grad_(need_grad)
            self.sh_params.requires_grad_(need_grad)
            self.rotate_vectors.requires_grad_(need_grad)
            self.positions.requires_grad_(need_grad)
            return True

        self.mask_params[anchor_mask].requires_grad_(need_grad)
        self.sh_params[anchor_mask].requires_grad_(need_grad)
        self.rotate_vectors[anchor_mask].requires_grad_(need_grad)
        self.positions[anchor_mask].requires_grad_(need_grad)
        return True

    def clearGrads(self, anchor_mask: Union[torch.Tensor, None] = None) -> bool:
        if anchor_mask is not None:
            if self.mask_params.grad is not None:
                self.mask_params.grad = None
            if self.sh_params.grad is not None:
                self.sh_params.grad = None
            if self.rotate_vectors.grad is not None:
                self.rotate_vectors.grad = None
            if self.positions.grad is not None:
                self.positions.grad = None
            return True

        if self.mask_params[anchor_mask].grad is not None:
            self.mask_params[anchor_mask].grad = None
        if self.sh_params[anchor_mask].grad is not None:
            self.sh_params[anchor_mask].grad = None
        if self.rotate_vectors[anchor_mask].grad is not None:
            self.rotate_vectors[anchor_mask].grad = None
        if self.positions[anchor_mask].grad is not None:
            self.positions[anchor_mask].grad = None
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

    @abstractmethod
    def updatePreLoadDatas(self) -> bool:
        pass

    def regularRotateVectors(self) -> bool:
        regular_rotate_vectors = toRegularRotateVectors(self.rotate_vectors)
        self.rotate_vectors.data = regular_rotate_vectors.detach().clone().type(self.dtype).to(self.device)
        return True

    def loadParams(
        self,
        mask_params: Union[torch.Tensor, np.ndarray, None] = None,
        sh_params: Union[torch.Tensor, np.ndarray, None] = None,
        rotate_vectors: Union[torch.Tensor, np.ndarray, None] = None,
        positions: Union[torch.Tensor, np.ndarray, None] = None,
        use_inv: Union[bool, None] = None,
        face_forward_vectors: Union[torch.Tensor, np.ndarray, None] = None,
        ortho6d_poses: Union[torch.Tensor, np.ndarray, None] = None,
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

            self.regularRotateVectors()

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

            trans_rotate_vectors = mash_cpp.toRotateVectorsByFaceForwardVectors(
                face_forward_vectors
            )

            self.rotate_vectors.data = (
                trans_rotate_vectors.detach().clone().type(self.dtype).to(self.device)
            )

            self.regularRotateVectors()

        if ortho6d_poses is not None:
            if not checkShape(ortho6d_poses.shape, [self.anchor_num, 6]):
                print("[ERROR][Mash::loadParams]")
                print("\t checkShape failed for ortho6d poses!")
                return False

            if isinstance(ortho6d_poses, np.ndarray):
                ortho6d_poses = torch.from_numpy(ortho6d_poses)

            rotate_vectors = toRotateVectorsFromOrthoPoses(ortho6d_poses)

            self.rotate_vectors.data = (
                rotate_vectors.detach().clone().type(self.dtype).to(self.device)
            )
        return True

    def loadParamsDict(self, params_dict: dict) -> bool:
        if 'mask_params' not in params_dict.keys():
            print("[ERROR][Mash::loadParamsDict]")
            print("\t mask_params not in params dict!")
            return False

        if 'sh_params' not in params_dict.keys():
            print("[ERROR][Mash::loadParamsDict]")
            print("\t sh_params not in params dict!")
            return False

        if 'rotate_vectors' not in params_dict.keys():
            print("[ERROR][Mash::loadParamsDict]")
            print("\t rotate_vectors not in params dict!")
            return False

        if 'positions' not in params_dict.keys():
            print("[ERROR][Mash::loadParamsDict]")
            print("\t positions not in params dict!")
            return False

        if 'use_inv' not in params_dict.keys():
            print("[ERROR][Mash::loadParamsDict]")
            print("\t use_inv not in params dict!")
            return False

        mask_params = params_dict["mask_params"]
        sh_params = params_dict["sh_params"]
        rotate_vectors = params_dict["rotate_vectors"]
        positions = params_dict["positions"]
        use_inv = params_dict["use_inv"]

        if not self.loadParams(mask_params, sh_params, rotate_vectors, positions, use_inv):
            print("[ERROR][Mash::loadParamsDict]")
            print("\t loadParams failed!")
            return False

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

    def mergeMash(self, mash) -> bool:
        if self.mask_degree_max != mash.mask_degree_max:
            print('[ERROR][BaseMash::mergeMash]')
            print('\t mask degree max not matched!')
            print('\t ', self.mask_degree_max, '!=', mash.mask_degree_max)
            return False

        if self.sh_degree_max != mash.sh_degree_max:
            print('[ERROR][BaseMash::mergeMash]')
            print('\t sh degree max not matched!')
            print('\t ', self.sh_degree_max, '!=', mash.sh_degree_max)
            return False

        self.setGradState(False)

        self.anchor_num += mash.anchor_num
        self.mask_params = torch.vstack([self.mask_params, mash.mask_params])
        self.sh_params = torch.vstack([self.sh_params, mash.sh_params])
        self.positions = torch.vstack([self.positions, mash.positions])
        self.rotate_vectors = torch.vstack([self.rotate_vectors, mash.rotate_vectors])

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

        new_mask_params, new_sh_params, new_rotate_vectors, new_positions = (
            toParams(
                self.anchor_num,
                self.mask_degree_max,
                self.sh_degree_max,
                self.dtype,
                self.device,
            )
        )

        new_mask_params[:copy_dim, :] = self.mask_params
        new_sh_params[:copy_dim, :] = self.sh_params
        new_positions[:copy_dim, :] = self.positions
        new_rotate_vectors[:copy_dim, :] = self.rotate_vectors

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

        new_mask_params = torch.zeros(
            [self.anchor_num, new_dim], dtype=self.dtype
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
            [self.anchor_num, new_dim], dtype=self.dtype
        ).to(self.device)

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

        move_position = move_position.type(self.positions.dtype).to(self.positions.device)

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

    def toOrtho6DPoses(self) -> torch.Tensor:
        ortho6d_poses = toOrthoPosesFromRotateVectors(self.rotate_vectors)
        return ortho6d_poses

    def toFaceToPoints(self) -> torch.Tensor:
        face_to_points = mash_cpp.toFaceToPoints(self.mask_degree_max, self.sh_degree_max, self.sh_params, self.rotate_vectors, self.positions, self.use_inv)
        return face_to_points

    def toWeightedSamplePoints(self, sample_phis: torch.Tensor, sample_theta_weights: torch.Tensor, sample_idxs: torch.Tensor, sample_base_values: torch.Tensor=torch.Tensor()) -> torch.Tensor:
        weighted_sample_points = mash_cpp.toWeightedSamplePoints(
            self.mask_degree_max, self.sh_degree_max, self.mask_params, self.sh_params, self.rotate_vectors,
            self.positions, sample_phis, sample_theta_weights,
            sample_idxs, self.use_inv, sample_base_values)

        return weighted_sample_points

    def toForceSamplePoints(self, sample_phis: torch.Tensor, sample_thetas: torch.Tensor, sample_idxs: torch.Tensor, sample_base_values: torch.Tensor=torch.Tensor()) -> torch.Tensor:
        sample_points = mash_cpp.toSamplePoints(
            self.mask_degree_max, self.sh_degree_max, self.sh_params, self.rotate_vectors, self.positions,
            sample_phis, sample_thetas, sample_idxs, self.use_inv,
            sample_base_values)
        return sample_points

    def toFPSPointIdxs(self, sample_points: torch.Tensor, sample_idxs: torch.Tensor, sample_point_scale: float) -> torch.Tensor:
        fps_sample_point_idxs = mash_cpp.toFPSPointIdxs(
            sample_points, sample_idxs, sample_point_scale, self.anchor_num
        )
        return fps_sample_point_idxs

    @abstractmethod
    def toSamplePointsWithNormals(self, refine_normals: bool=False, fps_sample_scale: float = -1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def toSamplePoints(self, with_normals: bool = False, refine_normals: bool = False, fps_sample_scale: float = -1) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        pass

    def toSamplePcdWithNormals(self, refine_normals: bool = False, fps_sample_scale: float = -1) -> o3d.geometry.PointCloud:
        mask_boundary_sample_points, in_mask_sample_points, in_mask_sample_point_idxs, mask_boundary_normals, in_mask_normals = self.toSamplePointsWithNormals(refine_normals, fps_sample_scale)

        sample_points = torch.vstack([mask_boundary_sample_points, in_mask_sample_points])
        sample_normals = torch.vstack([mask_boundary_normals, in_mask_normals])

        sample_points_array = toNumpy(sample_points)
        sample_normals_array = toNumpy(sample_normals)

        sample_pcd = getPointCloud(sample_points_array, sample_normals_array)
        return sample_pcd


    def toSamplePcd(self, with_normals: bool = False, refine_normals: bool = False, fps_sample_scale: float = -1) -> o3d.geometry.PointCloud:
        if with_normals:
            return self.toSamplePcdWithNormals(refine_normals, fps_sample_scale)

        mask_boundary_sample_points, in_mask_sample_points = self.toSamplePoints()[:2]

        sample_points = torch.vstack([mask_boundary_sample_points, in_mask_sample_points])

        sample_points_array = toNumpy(sample_points)

        sample_pcd = getPointCloud(sample_points_array)
        return sample_pcd

    def renderSamplePointsWithNormals(self, refine_normals: bool = False, fps_sample_scale: float = -1) -> bool:
        (
            mask_boundary_sample_points,
            in_mask_sample_points,
            in_mask_sample_point_idxs,
            valid_mask_boundary_normals,
            valid_in_mask_normals
        ) = self.toSamplePointsWithNormals(refine_normals, fps_sample_scale)

        boundary_pts = toNumpy(mask_boundary_sample_points)
        inner_pts = toNumpy(in_mask_sample_points)
        inner_anchor_idxs = toNumpy(in_mask_sample_point_idxs)
        boundary_normals = toNumpy(valid_mask_boundary_normals)
        inner_normals = toNumpy(valid_in_mask_normals)

        print("boundary_pts:", boundary_pts.shape, boundary_pts.dtype)
        print("inner_pts:", inner_pts.shape, inner_pts.dtype)
        print("inner_anchor_idxs:", inner_anchor_idxs.shape, inner_anchor_idxs.dtype)
        print('valid boundary_normals num:', torch.where(torch.norm(valid_mask_boundary_normals, dim=1) == 0)[0].shape)
        print('valid inner_normals num:', torch.where(torch.norm(valid_in_mask_normals, dim=1) == 0)[0].shape)

        if False:
            render_pcd_list = []

            for i in range(self.anchor_num):
                boundary_mask = self.mask_boundary_phi_idxs == i
                inner_mask = inner_anchor_idxs == i

                anchor_boundary_pts = boundary_pts[boundary_mask]
                anchor_inner_pts = inner_pts[inner_mask]
                anchor_boundary_normals = boundary_normals[boundary_mask]
                anchor_inner_normals = inner_normals[inner_mask]

                anchor_pts = np.vstack([anchor_boundary_pts, anchor_inner_pts])
                anchor_normals = np.vstack([anchor_boundary_normals, anchor_inner_normals]) * -1.0

                center = np.mean(anchor_pts, axis=0)

                anchor_pts = (
                    anchor_pts - center + 0.1 * np.array([i // 20, i % 20, 0.0])
                )

                pcd = getPointCloud(anchor_pts, anchor_normals)

                render_pcd_list.append(pcd)

            renderGeometries(render_pcd_list, 'Mash Anchor Sample Points With Normals', True)
            exit()

        sample_pts = np.vstack([inner_pts, boundary_pts])
        sample_normals = np.vstack([inner_normals, boundary_normals])
        sample_pts = inner_pts
        sample_normals = inner_normals 
        pcd = getPointCloud(sample_pts, sample_normals)
        renderGeometries(pcd, 'Mash Sample Points With Normals', True)
        return True

    def renderSamplePoints(self, with_normals: bool = False, refine_normals: bool = False, fps_sample_scale: float = -1) -> bool:
        if with_normals:
            return self.renderSamplePointsWithNormals(refine_normals, fps_sample_scale)

        (
            mask_boundary_sample_points,
            in_mask_sample_points,
            in_mask_sample_point_idxs,
        ) = self.toSamplePoints()

        boundary_pts = toNumpy(mask_boundary_sample_points)
        inner_pts = toNumpy(in_mask_sample_points)
        inner_anchor_idxs = toNumpy(in_mask_sample_point_idxs)

        print("boundary_pts:", boundary_pts.shape, boundary_pts.dtype)
        print("inner_pts:", inner_pts.shape, inner_pts.dtype)
        print("inner_anchor_idxs:", inner_anchor_idxs.shape, inner_anchor_idxs.dtype)

        if False:
            render_pcd_list = []

            for i in range(self.anchor_num):
                boundary_mask = self.mask_boundary_phi_idxs == i
                inner_mask = inner_anchor_idxs == i

                anchor_boundary_pts = boundary_pts[boundary_mask]
                anchor_inner_pts = inner_pts[inner_mask]

                anchor_pts = np.vstack([anchor_boundary_pts, anchor_inner_pts])

                center = np.mean(anchor_pts, axis=0)

                anchor_pts = (
                    anchor_pts - center + 0.1 * np.array([i // 20, i % 20, 0.0])
                )

                pcd = getPointCloud(anchor_pts)

                render_pcd_list.append(pcd)

            renderGeometries(render_pcd_list)
            exit()

        sample_pts = np.vstack([inner_pts, boundary_pts])
        sample_pts = inner_pts
        pcd = getPointCloud(sample_pts)
        renderGeometries(pcd)
        return True

    def renderSamplePatches(self) -> bool:
        boundary_pts, inner_pts, inner_idxs = self.toSamplePoints()
        sample_pts = torch.vstack([boundary_pts, inner_pts])
        sample_points = toNumpy(sample_pts)

        pcd = getPointCloud(sample_points)
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(4)

        if True:
            o3d.visualization.draw_geometries([pcd], point_show_normal=True)

        # FIXME: this algo looks not too bad, can be used to extract outer points
        if True:
            alpha = 0.03
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha
            )
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

        # FIXME: this algo looks bad, can only solve surface points, no inner point allowed
        if True:
            with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug
            ) as cm:
                mesh, densities = (
                    o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pcd, depth=9
                    )
                )

            render_mesh = deepcopy(mesh)
            render_mesh.compute_triangle_normals()
            render_mesh.compute_vertex_normals()

            densities = np.asarray(densities)
            density_colors = plt.get_cmap("plasma")(
                (densities - densities.min()) / (densities.max() - densities.min())
            )
            density_colors = density_colors[:, :3]
            density_mesh = o3d.geometry.TriangleMesh()
            density_mesh.vertices = mesh.vertices
            density_mesh.triangles = mesh.triangles
            density_mesh.triangle_normals = mesh.triangle_normals
            density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)

            density_mesh.translate([0, 0, -1])
            render_mesh.translate([0, 0, 1])

            renderGeometries([density_mesh, pcd, render_mesh])
        return True

    def toParamsDict(self) -> dict:
        params_dict = {
            "mask_params": toNumpy(self.mask_params),
            "sh_params": toNumpy(self.sh_params),
            "rotate_vectors": toNumpy(self.rotate_vectors),
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

        tmp_save_pcd_file_path = save_pcd_file_path[:-4] + "_tmp" + save_pcd_file_path[-4:]

        o3d.io.write_point_cloud(
            tmp_save_pcd_file_path, pcd, write_ascii=True, print_progress=print_progress
        )

        renameFile(tmp_save_pcd_file_path, save_pcd_file_path)
        return True
