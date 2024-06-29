import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import trange
from math import sqrt
from copy import deepcopy
from typing import Union, Tuple

import mash_cpp

from ma_sh.Config.degree import MAX_MASK_DEGREE, MAX_SH_DEGREE
from ma_sh.Data.mesh import Mesh
from ma_sh.Method.data import toNumpy
from ma_sh.Method.check import checkShape
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Method.center import toOuterCenters
from ma_sh.Method.rotate import toTriangleRotateMatrixs
from ma_sh.Method.Mash.mash import toParams
from ma_sh.Method.render import getCircle, renderGeometries
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
        self.sample_base_values = torch.tensor([0.0], dtype=dtype).to(self.device)
        self.mask_boundary_phi_idxs = torch.tensor([0.0], dtype=dtype).to(self.device)

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
        self.sample_phis = 2.0 * np.pi / self.sample_phi_num * torch.arange(self.sample_phi_num, dtype=self.dtype).to(self.device)
        self.sample_base_values = mash_cpp.toMaskBaseValues(self.sample_phis, self.mask_degree_max)
        self.mask_boundary_phi_idxs = torch.arange(self.anchor_num, dtype=self.idx_dtype).to(self.device).repeat(self.sample_phi_num, 1).permute(1, 0).reshape(-1)
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

    def toForceSamplePoints(self, sample_phis: torch.Tensor, sample_thetas: torch.Tensor, sample_idxs: torch.Tensor, sample_base_values: torch.Tensor=torch.Tensor()) -> torch.Tensor:
        sample_points = mash_cpp.toSimpleSamplePoints(
            self.mask_degree_max, self.sh_degree_max, self.sh_params, self.rotate_vectors, self.positions,
            sample_phis, sample_thetas, sample_idxs, self.use_inv,
            sample_base_values)
        return sample_points

    def toSimpleSamplePoints(self) -> torch.Tensor:
        simple_sample_points = mash_cpp.toSimpleMashSamplePoints(
            self.anchor_num,
            self.mask_degree_max,
            self.sh_degree_max,
            self.mask_params,
            self.sh_params,
            self.rotate_vectors,
            self.positions,
            self.sample_phis,
            self.sample_base_values,
            self.sample_theta_num,
            self.use_inv
        )

        return simple_sample_points

    def toSamplePoints(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        simple_sample_points = self.toSimpleSamplePoints()

        single_anchor_in_mask_point_num = 1 + self.sample_phi_num * (self.sample_theta_num - 1)
        single_anchor_point_num = single_anchor_in_mask_point_num + self.sample_phi_num

        mask_boundary_point_mask = torch.zeros(simple_sample_points.shape[0], dtype=torch.bool)
        for i in range(self.anchor_num):
            mask_boundary_point_mask[i * single_anchor_point_num + single_anchor_in_mask_point_num: (i + 1) * single_anchor_point_num] = True

        in_mask_points = simple_sample_points[~mask_boundary_point_mask]
        mask_boundary_points = simple_sample_points[mask_boundary_point_mask]
        in_mask_point_idxs = torch.arange(self.anchor_num, dtype=self.idx_dtype).to(self.device).repeat(single_anchor_in_mask_point_num, 1).permute(1, 0).reshape(-1)

        return mask_boundary_points, in_mask_points, in_mask_point_idxs

    def toSimpleSampleTriangles(self) -> np.ndarray:
        simple_sample_triangles = []

        single_anchor_point_num = 1 + self.sample_phi_num * self.sample_theta_num

        single_anchor_triangles = []

        for point_idx in range(1, self.sample_phi_num + 1):
            next_point_idx = point_idx % self.sample_phi_num + 1
            single_anchor_triangles.append([0, point_idx, next_point_idx])

        for cycle_idx in range(self.sample_theta_num - 1):
            point_idx_start = 1 + self.sample_phi_num * cycle_idx

            for j in range(self.sample_phi_num):
                point_idx = point_idx_start + j
                next_point_idx = point_idx_start + (j + 1) % self.sample_phi_num
                single_anchor_triangles.append([point_idx, point_idx + self.sample_phi_num, next_point_idx + self.sample_phi_num])
                single_anchor_triangles.append([point_idx, next_point_idx + self.sample_phi_num, next_point_idx])

        single_anchor_triangles = np.asarray(single_anchor_triangles, dtype=np.int64)

        for i in range(self.anchor_num):
            simple_sample_triangles.append(i * single_anchor_point_num + single_anchor_triangles)

        return np.vstack(simple_sample_triangles, dtype=np.int64)

    def toSimpleSampleSurfels(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points = self.toSimpleSamplePoints()
        triangles = self.toSimpleSampleTriangles()

        centers, radius = toOuterCenters(points, triangles)

        triangle_rotate_matrixs = toTriangleRotateMatrixs(points, triangles)

        return centers, radius, triangle_rotate_matrixs

    def toSimpleSampleCircles(self) -> list:
        centers, radius, triangle_rotate_matrixs = self.toSimpleSampleSurfels()

        centers = centers.detach().clone().cpu().numpy()
        radius = radius.detach().clone().cpu().numpy()
        triangle_rotate_matrixs = triangle_rotate_matrixs.detach().clone().cpu().numpy()

        circles = []

        for i in trange(centers.shape[0]):
            circle = getCircle(radius[i], 10)

            circle.rotate(triangle_rotate_matrixs[i])
            circle.translate(centers[i])

            circles.append(circle)

        return circles

    def toSampleMesh(self) -> Mesh:
        sample_mesh = Mesh()
        sample_mesh.vertices = self.toSimpleSamplePoints().detach().clone().cpu().numpy()
        sample_mesh.triangles = self.toSimpleSampleTriangles()
        return sample_mesh

    def toSamplePcd(self) -> o3d.geometry.PointCloud:
        mask_boundary_sample_points, in_mask_sample_points, in_mask_sample_point_idxs = self.toSamplePoints()

        sample_points = torch.vstack([mask_boundary_sample_points, in_mask_sample_points])

        sample_points_array = toNumpy(sample_points)

        sample_pcd = getPointCloud(sample_points_array)
        return sample_pcd

    def renderSamplePoints(self) -> bool:
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
