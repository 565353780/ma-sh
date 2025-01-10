import os
import torch
import numpy as np
import open3d as o3d
from typing import Union
from math import ceil, sqrt

from pointnet_pp.Module.detector import Detector

from ma_sh.Method.pcd import getPointCloud, toMergedPcd
from ma_sh.Model.simple_mash import SimpleMash

class AnchorMarker(object):
    def __init__(self,
                 kmeans_center_npy_file_path: Union[str, None] = None,
                 device: str = 'cpu',
                 ) -> None:
        self.device = device

        self.kmeans_centers = np.ndarray([0])
        self.kmeans_colors = np.ndarray([0])

        if kmeans_center_npy_file_path is not None:
            self.loadKMeansCenters(kmeans_center_npy_file_path)

        model_file_path = "../pointnet-pp/pretrained/cls_ssg/best_model.pth"
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
        shape_mash = mash.clone()

        shape_mash.loadParams(
            rotate_vectors=torch.zeros_like(shape_mash.rotate_vectors),
            positions=torch.zeros_like(shape_mash.positions),
        )

        inner_pts = shape_mash.toSamplePoints()[1]
        inner_pts = inner_pts.reshape(shape_mash.anchor_num, -1, 3).permute(0, 2, 1)

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

    def toAverageAnchors(self, mash: SimpleMash) -> SimpleMash:
        copy_mash = mash.clone()

        anchor_label = self.markMash(copy_mash)

        label_num = np.max(anchor_label) + 1

        new_positions = np.zeros([label_num, 3])
        average_mask_params = torch.zeros([label_num, copy_mash.mask_params.shape[1]], dtype=copy_mash.dtype, device=copy_mash.device)
        average_sh_params = torch.zeros([label_num, copy_mash.sh_params.shape[1]], dtype=copy_mash.dtype, device=copy_mash.device)

        col_num = ceil(sqrt(label_num))

        for i in range(label_num):
            row_idx = int(i / col_num)
            col_idx = i % col_num

            new_positions[i, 0] = 0.1 * row_idx
            new_positions[i, 1] = 0.1 * col_idx

            average_mask_params[i] = torch.mean(copy_mash.mask_params[anchor_label == i], dim=0)
            average_sh_params[i] = torch.mean(copy_mash.sh_params[anchor_label == i], dim=0)

        copy_mash.updateAnchorNum(label_num)

        anchor_label = np.arange(label_num)

        copy_mash.loadParams(
            average_mask_params,
            average_sh_params,
            torch.zeros_like(copy_mash.rotate_vectors),
            new_positions,
        )

        return copy_mash

    def toAnchorClusters(self, mash: SimpleMash, cluster_dist: float = 0.1) -> SimpleMash:
        copy_mash = mash.clone()

        anchor_label = self.markMash(copy_mash)

        new_positions = torch.zeros_like(copy_mash.positions)
        label_num = np.max(anchor_label) + 1

        col_num = ceil(sqrt(label_num))

        for i in range(label_num):
            row_idx = int(i / col_num)
            col_idx = i % col_num

            new_positions[anchor_label == i, 0] = cluster_dist * row_idx
            new_positions[anchor_label == i, 1] = cluster_dist * col_idx

        copy_mash.loadParams(
            rotate_vectors=torch.zeros_like(copy_mash.rotate_vectors),
            positions=new_positions,
        )

        return copy_mash

    def toReplacedAnchorMash(self, mash: SimpleMash) -> SimpleMash:
        anchor_label = self.markMash(mash)

        average_anchors = self.toAverageAnchors(mash)

        copy_mash = mash.clone()

        new_mask_params = torch.zeros_like(copy_mash.mask_params)
        new_sh_params = torch.zeros_like(copy_mash.sh_params)

        for i in range(average_anchors.anchor_num):
            new_mask_params[anchor_label == i, :] = average_anchors.mask_params[i]
            new_sh_params[anchor_label == i, :] = average_anchors.sh_params[i]

        copy_mash.loadParams(
            new_mask_params,
            new_sh_params,
        )

        return copy_mash

    def toPaintedMashPcd(self, mash: SimpleMash, anchor_label: np.ndarray) -> o3d.geometry.PointCloud:
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

        merged_pcd = toMergedPcd(boundary_pcd, inner_pcd)

        return merged_pcd

    def markAndRenderMashFile(self, mash_file_path: str) -> bool:
        if not os.path.exists(mash_file_path):
            print('[ERROR][AnchorMarker::markAndRenderMashFile]')
            print('\t mash file not exist!')
            print('\t mash_file_path:', mash_file_path)
            return False

        mash = SimpleMash.fromParamsFile(
            mash_file_path,
            sample_phi_num=20,
            sample_theta_num=20,
            device=self.device)

        anchor_label = self.markMash(mash)

        painted_mash_pcd = self.toPaintedMashPcd(mash, anchor_label)

        o3d.visualization.draw_geometries([painted_mash_pcd])

        return True

    def markAndRenderAnchorClusters(self, mash_file_path: str, cluster_dist: float = 0.1) -> bool:
        if not os.path.exists(mash_file_path):
            print('[ERROR][AnchorMarker::markAndRenderAnchorClusters]')
            print('\t mash file not exist!')
            print('\t mash_file_path:', mash_file_path)
            return False

        mash = SimpleMash.fromParamsFile(
            mash_file_path,
            sample_phi_num=20,
            sample_theta_num=20,
            device=self.device)

        anchor_label = self.markMash(mash)

        anchor_clusters = self.toAnchorClusters(mash, cluster_dist)

        painted_mash_pcd = self.toPaintedMashPcd(anchor_clusters, anchor_label)

        o3d.visualization.draw_geometries([painted_mash_pcd])

        return True

    def markAndRenderAverageAnchors(self, mash_file_path: str) -> bool:
        if not os.path.exists(mash_file_path):
            print('[ERROR][AnchorMarker::markAndRenderAverageAnchors]')
            print('\t mash file not exist!')
            print('\t mash_file_path:', mash_file_path)
            return False

        mash = SimpleMash.fromParamsFile(
            mash_file_path,
            sample_phi_num=20,
            sample_theta_num=20,
            device=self.device)

        average_anchors = self.toAverageAnchors(mash)

        anchor_label = np.arange(average_anchors.anchor_num)

        painted_mash_pcd = self.toPaintedMashPcd(average_anchors, anchor_label)

        o3d.visualization.draw_geometries([painted_mash_pcd])

        return True

    def markAndRenderMashReplacedByAverageAnchors(self, mash_file_path: str) -> bool:
        if not os.path.exists(mash_file_path):
            print('[ERROR][AnchorMarker::markAndRenderMashReplacedByAverageAnchors]')
            print('\t mash file not exist!')
            print('\t mash_file_path:', mash_file_path)
            return False

        mash = SimpleMash.fromParamsFile(
            mash_file_path,
            sample_phi_num=20,
            sample_theta_num=20,
            device=self.device)

        anchor_label = self.markMash(mash)

        replaced_anchor_mash = self.toReplacedAnchorMash(mash)

        painted_mash_pcd = self.toPaintedMashPcd(replaced_anchor_mash, anchor_label)

        o3d.visualization.draw_geometries([painted_mash_pcd])

        return True
