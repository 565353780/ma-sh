import torch
import numpy as np
from tqdm import trange
from math import sqrt
from typing import Union, Tuple

import mash_cpp

from ma_sh.Data.mesh import Mesh
from ma_sh.Model.base_mash import BaseMash
from ma_sh.Method.outer import toOuterCircles, toOuterEllipses
from ma_sh.Method.render import getCircle, getEllipse


class SimpleMash(BaseMash):
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
        self.sample_phi_num = sample_phi_num
        self.sample_theta_num = sample_theta_num

        # Pre Load Datas
        self.sample_phis = torch.tensor([0.0], dtype=dtype).to(device)
        self.sample_base_values = torch.tensor([0.0], dtype=dtype).to(device)
        self.mask_boundary_phi_idxs = torch.tensor([0.0], dtype=dtype).to(device)

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
        sample_phi_num: Union[int, None] = None,
        sample_theta_num: Union[int, None] = None,
        use_inv: Union[bool, None] = None,
        idx_dtype=None,
        dtype=None,
        device: Union[str, None] = None,
    ):
        mash = SimpleMash(
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

    def updatePreLoadDatas(self) -> bool:
        self.sample_phis = (
            2.0
            * np.pi
            / self.sample_phi_num
            * torch.arange(self.sample_phi_num, dtype=self.dtype).to(self.device)
        )
        self.sample_base_values = mash_cpp.toMaskBaseValues(
            self.sample_phis, self.mask_degree_max
        )
        self.mask_boundary_phi_idxs = (
            torch.arange(self.anchor_num, dtype=self.idx_dtype)
            .to(self.device)
            .repeat(self.sample_phi_num, 1)
            .permute(1, 0)
            .reshape(-1)
        )
        return True

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
            self.use_inv,
        )

        return simple_sample_points

    def toSamplePointsWithNormals(
        self, refine_normals: bool = False, fps_sample_scale: float = -1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        print("[ERROR][SimpleMash::toSamplePointsWithNormals]")
        print("\t this function has not been implemented!")
        return (
            torch.empty([0], dtype=self.dtype, device=self.device),
            torch.empty([0], dtype=self.dtype, device=self.device),
            torch.empty([0], dtype=self.dtype, device=self.device),
            torch.empty([0], dtype=self.dtype, device=self.device),
            torch.empty([0], dtype=self.dtype, device=self.device),
        )

    def toSamplePoints(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        simple_sample_points = self.toSimpleSamplePoints()

        single_anchor_in_mask_point_num = 1 + self.sample_phi_num * (
            self.sample_theta_num - 1
        )
        single_anchor_point_num = single_anchor_in_mask_point_num + self.sample_phi_num

        mask_boundary_point_mask = torch.zeros(
            simple_sample_points.shape[0], dtype=torch.bool
        )
        for i in range(self.anchor_num):
            mask_boundary_point_mask[
                i * single_anchor_point_num + single_anchor_in_mask_point_num : (i + 1)
                * single_anchor_point_num
            ] = True

        in_mask_points = simple_sample_points[~mask_boundary_point_mask]
        mask_boundary_points = simple_sample_points[mask_boundary_point_mask]
        in_mask_point_idxs = (
            torch.arange(self.anchor_num, dtype=self.idx_dtype)
            .to(self.device)
            .repeat(single_anchor_in_mask_point_num, 1)
            .permute(1, 0)
            .reshape(-1)
        )

        return mask_boundary_points, in_mask_points, in_mask_point_idxs

    def toSimpleSampleTriangles(self) -> np.ndarray:
        simple_sample_triangles = []

        if self.anchor_num == 0:
            print("[ERROR][SimpleMash::toSimpleSampleTriangles]")
            print("\t anchor is empty!")
            return np.asarray(simple_sample_triangles)

        single_anchor_point_num = 1 + self.sample_phi_num * self.sample_theta_num

        single_anchor_triangles = []

        for point_idx in range(1, self.sample_phi_num + 1):
            next_point_idx = point_idx % self.sample_phi_num + 1
            single_anchor_triangles.append([0, point_idx, next_point_idx])

        for cycle_idx in range(self.sample_theta_num - 1):
            if cycle_idx == 0:
                continue

            point_idx_start = 1 + self.sample_phi_num * cycle_idx

            for j in range(self.sample_phi_num):
                point_idx = point_idx_start + j
                next_point_idx = point_idx_start + (j + 1) % self.sample_phi_num
                single_anchor_triangles.append(
                    [
                        point_idx,
                        point_idx + self.sample_phi_num,
                        next_point_idx + self.sample_phi_num,
                    ]
                )
                single_anchor_triangles.append(
                    [point_idx, next_point_idx + self.sample_phi_num, next_point_idx]
                )

        single_anchor_triangles = np.asarray(single_anchor_triangles, dtype=np.int64)

        for i in range(self.anchor_num):
            simple_sample_triangles.append(
                i * single_anchor_point_num + single_anchor_triangles
            )

        return np.vstack(simple_sample_triangles, dtype=np.int64)

    def toSimpleSampleCircles(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points = self.toSimpleSamplePoints()
        triangles = self.toSimpleSampleTriangles()

        centers, radius = toOuterCircles(points, triangles)

        triangle_rotate_matrixs = toTriangleRotateMatrixs(points, triangles)

        return centers, radius, triangle_rotate_matrixs

    def toSimpleSampleEllipses(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points = self.toSimpleSamplePoints()
        triangles = self.toSimpleSampleTriangles()

        centers, axis_lengths, triangle_rotate_matrixs = toOuterEllipses(
            points, triangles
        )

        return centers, axis_lengths, triangle_rotate_matrixs

    def toSimpleSampleO3DCircles(self) -> list:
        centers, radius, triangle_rotate_matrixs = self.toSimpleSampleCircles()

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

    def toSimpleSampleO3DEllipses(self) -> list:
        centers, axis_lengths, rotate_matrixs = self.toSimpleSampleEllipses()

        centers = centers.detach().clone().cpu().numpy()
        axis_lengths = axis_lengths.detach().clone().cpu().numpy()
        rotate_matrixs = rotate_matrixs.detach().clone().cpu().numpy()

        ellipses = []

        for i in trange(centers.shape[0]):
            ellipse = getEllipse(axis_lengths[i][0], axis_lengths[i][1], 10)

            ellipse.rotate(rotate_matrixs[i])
            ellipse.translate(centers[i])

            ellipses.append(ellipse)

        return ellipses

    def toSampleMesh(self) -> Mesh:
        sample_mesh = Mesh()
        sample_mesh.vertices = (
            self.toSimpleSamplePoints().detach().clone().cpu().numpy()
        )
        sample_mesh.triangles = self.toSimpleSampleTriangles()
        return sample_mesh
