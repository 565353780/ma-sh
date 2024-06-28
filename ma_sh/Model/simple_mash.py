import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from math import sqrt
from copy import deepcopy
from torch.autograd import grad
from typing import Union, Tuple

from ma_sh.Method.Mash.mash_unit import toPreLoadBaseValues
import mash_cpp

from ma_sh.Config.degree import MAX_MASK_DEGREE, MAX_SH_DEGREE
from ma_sh.Data.mesh import Mesh
from ma_sh.Method.data import toNumpy
from ma_sh.Method.check import checkShape
from ma_sh.Method.normal import toNormalTags
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Method.Mash.mash import toParams, toPreLoadDatas
from ma_sh.Method.render import renderGeometries, renderPoints
from ma_sh.Method.path import createFileFolder, removeFile, renameFile


class SimpleMash(object):
    def __init__(
        self,
        anchor_num: int = 400,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        sample_phi_num: int = 10,
        sample_theta_num: int = 10,
        use_inv: bool = True,
        idx_dtype=torch.int64,
        dtype=torch.float32,
        device: str = "cpu",
    ) -> None:
        # Super Params
        self.anchor_num = anchor_num
        self.mask_degree_max = mask_degree_max
        self.sh_degree_max = sh_degree_max
        self.sample_phi_num = sample_phi_num
        self.sample_theta_num = sample_theta_num
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
        self.sample_base_values = torch.tensor([0.0], dtype=dtype).to(self.device)

        self.reset()
        return

    @classmethod
    def fromParamsDict(
        cls,
        params_dict: dict,
        sample_phi_num: int = 10,
        sample_theta_num: int = 10,
        idx_dtype=torch.int64,
        dtype=torch.float32,
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
            sample_phi_num,
            sample_theta_num,
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
        sample_phi_num: int = 10,
        sample_theta_num: int = 10,
        idx_dtype=torch.int64,
        dtype=torch.float32,
        device: str = "cpu",
    ):
        params_dict = np.load(params_file_path, allow_pickle=True).item()

        return cls.fromParamsDict(
            params_dict,
            sample_phi_num,
            sample_theta_num,
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

    def clearGrads(self) -> bool:
        if self.mask_params.grad is not None:
            self.mask_params.grad = None
        if self.sh_params.grad is not None:
            self.sh_params.grad = None
        if self.rotate_vectors.grad is not None:
            self.rotate_vectors.grad = None
        if self.positions.grad is not None:
            self.positions.grad = None
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
        single_zero_tensor = torch.zeros(1, dtype=self.dtype).to(self.device)
        one_circle_sample_phis = 2.0 * np.pi / self.sample_phi_num * torch.arange(self.sample_phi_num, dtype=self.dtype).to(self.device)
        self.sample_phis = torch.hstack([single_zero_tensor.clone(), one_circle_sample_phis.repeat(self.sample_theta_num)])

        one_circle_sample_thetas = np.pi / self.sample_theta_num * torch.arange(1, self.sample_theta_num + 1, dtype=self.dtype).to(self.device)
        self.sample_thetas = torch.hstack([single_zero_tensor.clone(), one_circle_sample_thetas.repeat(self.sample_phi_num, 1).permute(1, 0).reshape(-1)])

        self.sample_base_values = mash_cpp.toMaskBaseValues(self.sample_phis, self.mask_degree_max)
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
            print("[ERROR][SimpleMash::loadParams]")
            print("\t rotate vectors and face forward vectors are all None!")
            print("\t please make at least one of them be None!")
            return False

        if mask_params is not None:
            if not checkShape(mask_params.shape, self.mask_params.shape):
                print("[ERROR][SimpleMash::loadParams]")
                print("\t checkShape failed for mask params!")
                return False

            if isinstance(mask_params, np.ndarray):
                mask_params = torch.from_numpy(mask_params)

            self.mask_params.data = (
                mask_params.detach().clone().type(self.dtype).to(self.device)
            )

        if sh_params is not None:
            if not checkShape(sh_params.shape, self.sh_params.shape):
                print("[ERROR][SimpleMash::loadParams]")
                print("\t checkShape failed for sh params!")
                return False

            if isinstance(sh_params, np.ndarray):
                sh_params = torch.from_numpy(sh_params)

            self.sh_params.data = (
                sh_params.detach().clone().type(self.dtype).to(self.device)
            )

        if rotate_vectors is not None:
            if not checkShape(rotate_vectors.shape, self.rotate_vectors.shape):
                print("[ERROR][SimpleMash::loadParams]")
                print("\t checkShape failed for rotate vectors!")
                return False

            if isinstance(rotate_vectors, np.ndarray):
                rotate_vectors = torch.from_numpy(rotate_vectors)

            self.rotate_vectors.data = (
                rotate_vectors.detach().clone().type(self.dtype).to(self.device)
            )

        if positions is not None:
            if not checkShape(positions.shape, self.positions.shape):
                print("[ERROR][SimpleMash::loadParams]")
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
                print("[ERROR][SimpleMash::loadParams]")
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
            print("[ERROR][SimpleMash::loadParamsFile]")
            print("\t params dict file not exist!")
            print("\t params_file_path:", params_file_path)
            return False

        params_dict = np.load(params_file_path, allow_pickle=True).item()

        if not self.loadParamsDict(params_dict):
            print("[ERROR][SimpleMash::loadParamsFile]")
            print("\t loadParamsDict failed!")
            return False

        return True

    def updateMaskDegree(self, mask_degree_max: int) -> bool:
        if mask_degree_max == self.mask_degree_max:
            return True

        if mask_degree_max < 0 or mask_degree_max > MAX_MASK_DEGREE:
            print("[ERROR][SimpleMash::updateMaskDegree]")
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
            print("[ERROR][SimpleMash::updateSHDegree]")
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

    def toCenter(self) -> torch.Tensor:
        center = torch.mean(self.positions.data, dim=0).clone()
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

    def toSamplePoints(self) -> torch.Tensor:
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

        mask_boundary_sample_points, in_mask_sample_points, in_mask_sample_point_idxs = self.toSamplePoints()

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
        self, save_params_file_path: str, overwrite: bool = False
    ) -> bool:
        if os.path.exists(save_params_file_path):
            if overwrite:
                removeFile(save_params_file_path)
            else:
                print("[WARN][SimpleMash::saveParamsFile]")
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

    def saveAsPcdFile(
        self,
        save_pcd_file_path: str,
        overwrite: bool = False,
        print_progress: bool = False,
    ) -> bool:
        if os.path.exists(save_pcd_file_path):
            if overwrite:
                removeFile(save_pcd_file_path)
            else:
                print("[ERROR][SimpleMash::saveAsPcdFile]")
                print("\t save pcd file already exist!")
                print("\t save_pcd_file_path:", save_pcd_file_path)
                return False

        createFileFolder(save_pcd_file_path)

        boundary_pts, inner_pts, inner_idxs = self.toSamplePoints()
        sample_pts = torch.vstack([boundary_pts, inner_pts])
        sample_points = toNumpy(sample_pts)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sample_points)

        if print_progress:
            print("[INFO][SimpleMash::saveAsPcdFile]")
            print("\t start save as pcd file...")
        o3d.io.write_point_cloud(
            save_pcd_file_path, pcd, write_ascii=True, print_progress=print_progress
        )
        return True
