import os
import torch
import trimesh
import numpy as np
import open3d as o3d
from typing import Union, Tuple
from copy import deepcopy

import mash_cpp

from ma_sh.Data.point import Point
from ma_sh.Data.pcd import Pcd
from ma_sh.Method.data import toNumpy
from ma_sh.Method.io import loadMeshFile
from ma_sh.Method.path import createFileFolder, removeFile, renameFile
from ma_sh.Method.mesh import samplePointCloud, samplePoints
from ma_sh.Method.sort import getNearIdxs
from ma_sh.Method.color import getJetColorsFromDists
from ma_sh.Method.render import renderGeometries
from ma_sh.Module.timer import Timer


class Mesh(object):
    def __init__(
        self,
        mesh_file_path: Union[str, None] = None,
        sample_point_num: Union[int, None] = None,
    ) -> None:
        self.triangle_normals = None
        self.triangles = None
        self.vertex_colors = None
        self.vertex_normals = None
        self.vertices = None

        self.sample_pts = None
        self.sample_normals = None

        if mesh_file_path is not None:
            self.loadMesh(mesh_file_path)
        if sample_point_num is not None:
            self.samplePoints(sample_point_num)
        return

    def reset(self):
        self.mesh = None

        self.sample_pts = None
        self.sample_normals = None
        return True

    @classmethod
    def from_o3d(cls, o3d_mesh: o3d.geometry.TriangleMesh):
        mesh = cls()
        mesh.loadO3DMeshProperties(o3d_mesh)
        return mesh

    def loadO3DMeshProperties(self, o3d_mesh: o3d.geometry.TriangleMesh) -> bool:
        self.triangle_normals = np.asarray(o3d_mesh.triangle_normals)
        self.triangles = np.asarray(o3d_mesh.triangles)
        self.vertex_colors = np.asarray(o3d_mesh.vertex_colors)
        self.vertex_normals = np.asarray(o3d_mesh.vertex_normals)
        self.vertices = np.asarray(o3d_mesh.vertices)
        return True

    def loadMesh(self, mesh_file_path: str):
        self.reset()

        o3d_mesh = loadMeshFile(mesh_file_path)

        if o3d_mesh is None:
            print("[ERROR][Mesh::loadMesh]")
            print("\t loadMeshFile failed!")
            return False

        return self.loadO3DMeshProperties(o3d_mesh)

    def isValid(self, output_info=False):
        if self.vertices is None:
            if output_info:
                print("[ERROR][Mesh::isValid]")
                print("\t vertices is None! please load mesh first!")
            return False

        if self.triangles is None:
            if output_info:
                print("[ERROR][Mesh::isValid]")
                print("\t triangles is None! please load mesh first!")
            return False

        if self.vertices.shape[0] == 0:
            if output_info:
                print("[ERROR][Mesh::isValid]")
                print("\t vertices is empty! please check this mesh!")
            return False

        return True

    def center(self) -> Point:
        min_bound = np.min(self.vertices, axis=0)
        max_bound = np.max(self.vertices, axis=0)

        center = (min_bound + max_bound) / 2.0
        return Point.from_numpy(center)

    def length(self) -> float:
        min_bound = np.min(self.vertices, axis=0)
        max_bound = np.max(self.vertices, axis=0)
        length = np.max(max_bound - min_bound)
        return length

    def normalize(self) -> bool:
        center = self.center().numpy()
        scale = 0.9 / self.length()
        self.vertices = (self.vertices - center) * scale
        return True

    def points(self) -> np.ndarray:
        if self.vertices is None:
            print("[WARN][Mesh::points]")
            print("\t vertices is None!")
            return np.ndarray([])

        return self.vertices

    def pointNum(self) -> int:
        return self.points().shape[0]

    def samplePointNum(self) -> int:
        return self.sample_pts.shape[0]

    def pointNormals(self) -> Union[np.ndarray, None]:
        if self.vertex_normals is None:
            print("[WARN][Mesh::normals]")
            print("\t vertex_normals is None!")
            return None

        return np.asarray(self.vertex_normals)

    def faceNormals(self) -> Union[np.ndarray, None]:
        if self.triangle_normals is None:
            print("[WARN][Mesh::normals]")
            print("\t triangle_normals is None!")
            return None

        return np.asarray(self.triangle_normals)

    def colors(self) -> np.ndarray:
        if self.vertex_colors is None:
            print("[WARN][Mesh::colors]")
            print("\t vertex_colors is None!")
            return np.ndarray([])

        return np.asarray(self.vertex_colors)

    def facesPointIdxs(self) -> np.ndarray:
        if self.triangles is None:
            print("[WARN][Mesh::facesPointIdxs]")
            print("\t triangles is None!")
            return np.ndarray([])

        return np.asarray(self.triangles)

    def faceNum(self) -> int:
        return self.facesPointIdxs().shape[0]

    def paintJetColorsByDists(
        self, dists: Union[list, np.ndarray], error_max_percent: float = 0.1
    ) -> bool:
        dists = np.array(dists, dtype=float)

        if dists.shape[0] != self.vertices.shape[0]:
            print("[ERROR][Mesh::paintJetColorsByDists]")
            print("\t dists shape not matched with vertices!")
            print("\t dists.shape:", dists.shape)
            print("\t vertices.shape:", self.vertices.shape)
            return False

        min_dist = 0
        max_dist = self.toABBLength() * error_max_percent

        if min_dist == max_dist:
            scaled_dists = dists
        else:
            scaled_dists = (dists - min_dist) / (max_dist - min_dist)

        self.vertex_colors = getJetColorsFromDists(scaled_dists)
        return True

    def paintJetColorsByPoints(
        self,
        points: Union[list, np.ndarray],
        error_max_percent: float = 0.1,
        accurate: bool = True,
    ) -> bool:
        if accurate:
            dists = np.array(
                [
                    np.min(np.linalg.norm(vertex - points, ord=2, axis=1))
                    for vertex in self.vertices
                ],
                dtype=float,
            )
        else:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"

            torch_vertices = (
                torch.from_numpy(self.vertices)
                .type(torch.float32)
                .reshape(1, -1, 3)
                .to(device)
            )
            torch_points = (
                torch.from_numpy(points)
                .type(torch.float32)
                .reshape(1, -1, 3)
                .to(device)
            )
            torch_dists = mash_cpp.toChamferDistance(torch_vertices, torch_points)[0]
            dists = toNumpy(torch_dists).reshape(-1)

        return self.paintJetColorsByDists(dists, error_max_percent)

    def toO3DMesh(self) -> o3d.geometry.TriangleMesh:
        o3d_mesh = o3d.geometry.TriangleMesh()
        if self.triangle_normals is not None:
            o3d_mesh.triangle_normals = o3d.utility.Vector3dVector(
                self.triangle_normals
            )
        if self.triangles is not None:
            o3d_mesh.triangles = o3d.utility.Vector3iVector(self.triangles)
        if self.vertex_colors is not None:
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(self.vertex_colors)
        if self.vertex_normals is not None:
            o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(self.vertex_normals)
        if self.vertices is not None:
            o3d_mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        return o3d_mesh

    def toO3DTensorMesh(self) -> o3d.t.geometry.TriangleMesh:
        return o3d.t.geometry.TriangleMesh.from_legacy(self.toO3DMesh())

    def toTrimesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(
            vertices=deepcopy(self.vertices),
            faces=deepcopy(self.triangles),
            face_normals=deepcopy(self.triangle_normals),
            vertex_normals=deepcopy(self.vertex_normals),
        )

    def toFacePoints(self, face_idx: int) -> Union[np.ndarray, None]:
        if face_idx < 0 or face_idx >= self.faceNum():
            print("[ERROR][Mesh::toFacePoints]")
            print("\t face idx out of range!")
            print("\t face_idx:", face_idx, ">", self.faceNum())
            return None

        face_point_idxs = self.facesPointIdxs()[face_idx]
        return self.points()[face_point_idxs]

    def toFacesPoints(self) -> np.ndarray:
        return self.points()[self.facesPointIdxs()]

    def toFaceCenter(self, face_idx: int) -> Union[Point, None]:
        face_points = self.toFacePoints(face_idx)
        if face_points is None:
            print("[ERROR][Mesh::toFacePoints]")
            print("\t toFacePoints failed!")
            return None

        return Point.from_numpy(np.mean(face_points, axis=0))

    def toFaceCenters(self) -> Union[np.ndarray, None]:
        return np.mean(self.toFacesPoints(), axis=1)

    def toNearestSamplePointIdx(self, point: Point) -> int:
        return np.linalg.norm(self.sample_pts - point.numpy(), ord=2, axis=1).argsort()[
            0
        ]

    def toNearestSamplePoint(self, point: Point) -> Point:
        return Point.from_numpy(self.sample_pts[self.toNearestSamplePointIdx()])

    def toNearestPointIdx(self, point: Point) -> int:
        return np.linalg.norm(self.points() - point.numpy(), ord=2, axis=1).argsort()[0]

    def toNearestPoint(self, point: Point) -> Point:
        return Point.from_numpy(self.points()[self.toNearestPointIdx()])

    def toNearestFaceIdx(self, point: Point) -> int:
        return np.linalg.norm(
            self.toFaceCenters() - point.numpy(), ord=2, axis=1
        ).argsort()[0]

    def toNearestFaceCenter(self, point: Point) -> Point:
        return Point.from_numpy(self.toFaceCenters()[self.toNearestFaceIdx()])

    def toPcd(self, sample_point_num):
        if not self.isValid(True):
            print("[ERROR][Mesh::toPcd]")
            print("\t isValid failed!")
            return None

        new_pcd = samplePointCloud(self.toO3DMesh(), sample_point_num)

        if new_pcd is None:
            print("[ERROR][Mesh::toPcd]")
            print("\t samplePointCloud failed!")
            return None

        pcd = Pcd()
        pcd.pcd = new_pcd
        return pcd

    def toSamplePoints(self, sample_point_num: int, with_color: bool=False) -> Union[Tuple[None, None], None, Tuple[np.ndarray, np.ndarray], np.ndarray]:
        if not self.isValid(True):
            print("[ERROR][Mesh::toSamplePoints]")
            print("\t isValid failed!")
            if with_color:
                return None, None
            return None

        print("[INFO][Mesh::toSamplePoints]")
        print("\t start samplePoints with", sample_point_num, "points...")
        timer = Timer()
        sample_points = samplePoints(self.toO3DMesh(), sample_point_num, with_color)
        print(
            "\t samplePoints with",
            sample_point_num,
            "points finished! time:",
            timer.now(),
        )

        if with_color:
            if None in sample_points:
                print("[ERROR][Mesh::toSamplePoints]")
                print("\t samplePoins failed!")
                return None, None

        if sample_points is None:
            print("[ERROR][Mesh::toSamplePoints]")
            print("\t samplePoins failed!")
            return None

        return sample_points

    def toRandomSamplePoints(
        self,
        sample_point_num: int,
        random_weight: float = 0.0,
        noise_weight: float = 0.0,
        with_color: bool=False,
    ) -> Union[Tuple[None, None], None, Tuple[np.ndarray, np.ndarray], np.ndarray]:
        if not self.isValid(True):
            print("[ERROR][Mesh::toRandomSamplePoints]")
            print("\t isValid failed!")
            if with_color:
                return None, None
            return None

        if random_weight > 0:
            uniform_sample_point_num = int(sample_point_num * (1.0 + random_weight))
            ratio = sample_point_num / uniform_sample_point_num

            o3d_mesh = self.toO3DMesh()
            sample_pcd = o3d_mesh.sample_points_uniformly(uniform_sample_point_num)

            random_sample_pcd = sample_pcd.random_down_sample(ratio)

            random_sample_points = np.asarray(random_sample_pcd.points)
        else:
            random_sample_points = self.toO3DMesh().sample_points_uniformly(sample_point_num)

        if noise_weight > 0:
            noise = np.random.normal(scale=noise_weight, size=random_sample_points.shape)
            random_sample_points += noise

        if not with_color:
            return random_sample_points

        random_sample_colors = np.asarray(random_sample_pcd.colors)

        return random_sample_points, random_sample_colors

    def toNearSamplePointIdxs(self, point: Point) -> np.ndarray:
        return getNearIdxs(point.numpy(), self.sample_pts)

    def toNearestSamplePointIdxs(self, point: Point) -> int:
        return self.toNearSamplePointIdxs(point)[0]

    def toNearSamplePointIdxsWithMask(
        self, point: Point, sample_point_mask: np.ndarray
    ) -> np.ndarray:
        valid_sample_point_idxs = np.where(sample_point_mask == True)[0]
        valid_sample_points = self.sample_pts[valid_sample_point_idxs]
        near_valid_sample_point_idxs = getNearIdxs(point.numpy(), valid_sample_points)
        return valid_sample_point_idxs[near_valid_sample_point_idxs]

    def toNearestSamplePointIdxWithMask(
        self, point: Point, sample_point_mask: np.ndarray
    ) -> np.ndarray:
        return self.toNearSamplePointIdxsWithMask(point, sample_point_mask)[0]

    def toO3DABB(self) -> o3d.geometry.AxisAlignedBoundingBox:
        return self.toO3DMesh().get_axis_aligned_bounding_box()

    def toABBMaxBound(self) -> np.ndarray:
        return np.array(self.toO3DABB().get_max_bound(), dtype=np.float64)

    def toABBLength(self) -> float:
        return np.linalg.norm(self.toABBMaxBound(), ord=2)

    def toBoundaryIdxs(self) -> np.ndarray:
        half_edge_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(
            self.toO3DMesh()
        )
        return np.array(half_edge_mesh.get_boundaries(), dtype=int).reshape(-1)

    def toBoundaryPoints(self) -> np.ndarray:
        return self.vertices[self.toBoundaryIdxs()]

    def samplePoints(self, sample_point_num: int) -> bool:
        sample_pcd = samplePointCloud(self.toO3DMesh(), sample_point_num)
        if sample_pcd is None:
            print("[ERROR][Mesh::samplePoints]")
            print("\t samplePointCloud failed!")
            return False

        self.sample_pts = np.asarray(sample_pcd.points)
        self.sample_normals = np.asarray(sample_pcd.normals)
        return True

    def toSamplePcd(self) -> Pcd:
        sample_pcd = o3d.geometry.PointCloud()
        sample_pcd.points = o3d.utility.Vector3dVector(self.sample_pts)
        sample_pcd.normals = o3d.utility.Vector3dVector(self.sample_normals)
        colors = np.zeros([self.sample_pts.shape[0], 3], dtype=float)
        colors[:, 0] = 1
        sample_pcd.colors = o3d.utility.Vector3dVector(colors)
        return Pcd.from_o3d(sample_pcd)

    def save(self, save_mesh_file_path: str, overwrite: bool = False) -> bool:
        if not overwrite:
            if os.path.exists(save_mesh_file_path):
                return True

        removeFile(save_mesh_file_path)

        o3d_mesh = self.toO3DMesh()

        tmp_save_mesh_file_path = save_mesh_file_path[:-4] + '_tmp' + save_mesh_file_path[-4:]

        createFileFolder(save_mesh_file_path)

        o3d.io.write_triangle_mesh(tmp_save_mesh_file_path, o3d_mesh, write_ascii=True)

        renameFile(tmp_save_mesh_file_path, save_mesh_file_path)
        return True

    def render(self):
        if not self.isValid(True):
            print("[ERROR][Mesh::render]")
            print("\t isValid failed!")
            return False

        renderGeometries(self.toO3DMesh(), "Mesh")
        return True
