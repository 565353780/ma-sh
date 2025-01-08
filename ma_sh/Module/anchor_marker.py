import os
import numpy as np
import open3d as o3d
from typing import Union

from pointnet_pp.Module.detector import Detector

from ma_sh.Method.pcd import getPointCloud
from ma_sh.Model.simple_mash import SimpleMash

class AnchorMarker(object):
    def __init__(self,
                 kmeans_center_npy_file_path: Union[str, None] = None,
                 ) -> None:
        self.kmeans_centers = np.ndarray([0])
        self.kmeans_colors = np.ndarray([0])

        if kmeans_center_npy_file_path is not None:
            self.loadKMeansCenters(kmeans_center_npy_file_path)

        model_file_path = "../pointnet-pp/pretrained/cls_ssg/best_model.pth"
        self.device = "cpu"
        self.detector = Detector(model_file_path, self.device)
        return

    def loadKMeansCenters(self, kmeans_center_npy_file_path: str) -> bool:
        if not os.path.exists(kmeans_center_npy_file_path):
            print('[ERROR][AnchorMarker::loadKMeansCenters]')
            print('\t kmeans center npy file not exist!')
            print('\t kmeans_center_npy_file_path:', kmeans_center_npy_file_path)
            return False

        self.kmeans_centers = np.load(kmeans_center_npy_file_path)

        self.kmeans_colors = np.random.rand(self.kmeans_centers.shape[0], 3)
        return True

    def markMash(self, mash: SimpleMash) -> np.ndarray:
        inner_pts = mash.toSamplePoints()[1]
        inner_pts = inner_pts.reshape(mash.anchor_num, -1, 3).permute(0, 2, 1)

        _, feature = self.detector.detect(inner_pts)
        feature = feature.squeeze(-1).cpu().numpy()

        distances = np.linalg.norm(feature[:, np.newaxis, :] - self.kmeans_centers, axis=2)

        closest_indices = np.argmin(distances, axis=1)

        return closest_indices

    def markMashFile(self, mash_file_path: str) -> Union[np.ndarray, None]:
        if not os.path.exists(mash_file_path):
            print('[ERROR][AnchorMarker::markMashFile]')
            print('\t mash file not exist!')
            print('\t mash_file_path:', mash_file_path)
            return None

        mash = SimpleMash.fromParamsFile(mash_file_path, device=self.device)

        anchor_label = self.markMash(mash)

        return anchor_label

    def markAndRenderMashFile(self, mash_file_path: str) -> bool:
        if not os.path.exists(mash_file_path):
            print('[ERROR][AnchorMarker::markAndRenderMashFile]')
            print('\t mash file not exist!')
            print('\t mash_file_path:', mash_file_path)
            return False

        mash = SimpleMash.fromParamsFile(mash_file_path, device=self.device)

        anchor_label = self.markMash(mash)

        boundary_pts, inner_pts, inner_idxs = mash.toSamplePoints()
        inner_pts = inner_pts.detach().clone().cpu().numpy()
        boundary_pts = boundary_pts.detach().clone().cpu().numpy()
        inner_idxs = inner_idxs.cpu().numpy()
        boundary_idxs = mash.mask_boundary_phi_idxs.cpu().numpy()

        boundary_pcd = getPointCloud(boundary_pts)
        inner_pcd = getPointCloud(inner_pts)

        boundary_colors = self.kmeans_colors[anchor_label[boundary_idxs]]
        inner_colors = self.kmeans_colors[anchor_label[inner_idxs]]

        boundary_pcd.colors = o3d.utility.Vector3dVector(boundary_colors)
        inner_pcd.colors = o3d.utility.Vector3dVector(inner_colors)

        o3d.visualization.draw_geometries([boundary_pcd, inner_pcd])

        return True
