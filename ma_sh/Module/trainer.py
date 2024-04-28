import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Union
from copy import deepcopy
from torch.optim import AdamW

import mash_cpp

from ma_sh.Config.weights import W0
from ma_sh.Config.constant import EPSILON
from ma_sh.Config.degree import MAX_MASK_DEGREE, MAX_SH_DEGREE
from ma_sh.Data.mesh import Mesh
from ma_sh.Method.data import toNumpy
from ma_sh.Method.pcd import getPointCloud, downSample
from ma_sh.Method.time import getCurrentTime
from ma_sh.Model.mash import Mash
from ma_sh.Module.logger import Logger
from ma_sh.Module.o3d_viewer import O3DViewer


class Trainer(object):
    def __init__(
        self,
        anchor_num: int = 400,
        mask_degree_max: int = 4,
        sh_degree_max: int = 3,
        mask_boundary_sample_num: int = 10,
        sample_polar_num: int = 10000,
        sample_point_scale: float = 0.4,
        use_inv: bool = True,
        idx_dtype=torch.int64,
        dtype=torch.float64,
        device: str = "cpu",
        fit_lr: float = 5e-3,
        lr: float = 5e-3,
        fit_step_num: int = 20,
        train_epoch: int = 10,
        patience: int = 4,
        render: bool = False,
        render_freq: int = 1,
        render_init_only: bool = False,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
    ) -> None:
        self.mash = Mash(
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

        self.fit_lr = fit_lr
        self.lr = lr
        self.fit_step_num = fit_step_num
        self.train_epoch = train_epoch
        self.patience = patience

        self.epoch = 0
        self.step = 0
        self.loss_min = float("inf")

        self.best_params_dict = {}

        self.render = render
        self.render_freq = render_freq
        self.render_init_only = render_init_only

        self.save_result_folder_path = save_result_folder_path
        self.save_log_folder_path = save_log_folder_path
        self.save_file_idx = 0
        self.logger = Logger()

        self.mesh = Mesh()
        self.gt_points = None

        self.optimizer = AdamW(
            [
                self.mash.mask_params,
                self.mash.sh_params,
                self.mash.rotate_vectors,
                self.mash.positions,
            ],
            lr=self.lr,
        )

        self.initRecords()

        self.o3d_viewer = None
        if self.render:
            self.o3d_viewer = O3DViewer()
            self.o3d_viewer.createWindow()

        self.min_lr_reach_time = 0
        return

    def initRecords(self) -> bool:
        self.save_file_idx = 0

        current_time = getCurrentTime()

        if self.save_result_folder_path == "auto":
            self.save_result_folder_path = "./output/" + current_time + "/"
        if self.save_log_folder_path == "auto":
            self.save_log_folder_path = "./logs/" + current_time + "/"

        if self.save_result_folder_path is not None:
            os.makedirs(self.save_result_folder_path, exist_ok=True)
        if self.save_log_folder_path is not None:
            os.makedirs(self.save_log_folder_path, exist_ok=True)
            self.logger.setLogFolder(self.save_log_folder_path)
        return True

    def loadGTPointsFile(self, gt_points_file_path: str) -> bool:
        if not os.path.exists(gt_points_file_path):
            print("[ERROR][Trainer::loadGTPointsFile]")
            print("\t gt points file not exist!")
            print("\t gt_points_file_path:", gt_points_file_path)
            return False

        self.gt_points = np.load(gt_points_file_path)

        gt_pcd = getPointCloud(self.gt_points)
        gt_pcd.estimate_normals()

        surface_dist = 0.001

        anchor_pcd = downSample(gt_pcd, self.mash.anchor_num)

        if anchor_pcd is None:
            print("[ERROR][Trainer::loadGTPointsFile]")
            print("\t downSample failed!")
            return False

        sample_pts = np.asarray(anchor_pcd.points)
        sample_normals = np.asarray(anchor_pcd.normals)

        sh_params = torch.ones_like(self.mash.sh_params) * EPSILON
        sh_params[:, 0] = surface_dist / W0[0]

        self.mash.loadParams(
            sh_params=sh_params,
            positions=sample_pts + surface_dist * sample_normals,
            face_forward_vectors=-sample_normals,
        )
        return True

    def loadMeshFile(self, mesh_file_path: str) -> bool:
        if not os.path.exists(mesh_file_path):
            print("[ERROR][Trainer::loadMeshFile]")
            print("\t mesh file not exist!")
            print("\t mesh_file_path:", mesh_file_path)
            return False

        if not self.mesh.loadMesh(mesh_file_path):
            print("[ERROR][Trainer::loadMeshFile]")
            print("\t loadMesh failed!")
            print("\t mesh_file_path:", mesh_file_path)
            return False

        surface_dist = 0.001

        self.mesh.samplePoints(self.mash.anchor_num)

        assert self.mesh.sample_normals is not None
        assert self.mesh.sample_pts is not None

        sh_params = torch.zeros_like(self.mash.sh_params)
        sh_params[:, 0] = surface_dist / W0[0]

        self.mash.loadParams(
            sh_params=sh_params,
            positions=self.mesh.sample_pts + surface_dist * self.mesh.sample_normals,
            face_forward_vectors=-self.mesh.sample_normals,
        )
        return True

    def loadParams(
        self,
        mask_params: Union[torch.Tensor, np.ndarray],
        sh_params: Union[torch.Tensor, np.ndarray],
        rotate_vectors: Union[torch.Tensor, np.ndarray],
        positions: Union[torch.Tensor, np.ndarray],
        use_inv: bool,
    ) -> bool:
        self.mash.loadParams(mask_params, sh_params, rotate_vectors, positions, use_inv)
        return True

    def loadBestParams(
        self,
        mask_params: Union[torch.Tensor, np.ndarray],
        sh_params: Union[torch.Tensor, np.ndarray],
        rotate_vectors: Union[torch.Tensor, np.ndarray],
        positions: Union[torch.Tensor, np.ndarray],
        use_inv: bool,
    ) -> bool:
        if isinstance(mask_params, np.ndarray):
            mask_params = torch.from_numpy(mask_params)
        if isinstance(sh_params, np.ndarray):
            sh_params = torch.from_numpy(sh_params)
        if isinstance(rotate_vectors, np.ndarray):
            rotate_vectors = torch.from_numpy(rotate_vectors)
        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions)

        self.best_params_dict["mask_params"] = mask_params
        self.best_params_dict["sh_params"] = sh_params
        self.best_params_dict["rotate_vectors"] = rotate_vectors
        self.best_params_dict["positions"] = positions
        self.best_params_dict["use_inv"] = use_inv
        return True

    def upperMaskDegree(self) -> bool:
        if self.mash.mask_degree_max == MAX_MASK_DEGREE:
            return False

        if not self.mash.updateMaskDegree(self.mash.mask_degree_max + 1):
            print("[ERROR][Trainer::upperMaskDegree]")
            print("\t updateMaskDegree failed!")
            return False
        return True

    def upperSHDegree(self) -> bool:
        if self.mash.sh_degree_max == MAX_SH_DEGREE:
            return False

        if not self.mash.updateSHDegree(self.mash.sh_degree_max + 1):
            print("[ERROR][Trainer::upperMaskDegree]")
            print("\t updateSHDegree failed!")
            return False
        return True

    def updateLr(self, lr: float) -> bool:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return True

    def trainStep(
        self,
        gt_points: torch.Tensor,
    ) -> Union[dict, None]:
        self.optimizer.zero_grad()

        boundary_idxs = self.mash.mask_boundary_phi_idxs
        boundary_pts, inner_pts, inner_idxs = self.mash.toSampleUnitPoints()

        mash_pts = torch.vstack([boundary_pts, inner_pts])

        fit_dists2, coverage_dists2 = mash_cpp.toChamferDistance(
            mash_pts.unsqueeze(0).type(gt_points.dtype), gt_points
        )[:2]

        fit_dists = torch.sqrt(fit_dists2 + EPSILON)
        coverage_dists = torch.sqrt(coverage_dists2 + EPSILON)

        fit_loss = torch.mean(fit_dists)
        coverage_loss = torch.mean(coverage_dists)

        boundary_connect_loss = 0

        for i in range(self.mash.anchor_num):
            current_boundary_pts_mask = boundary_idxs == i
            current_boundary_pts = boundary_pts[current_boundary_pts_mask]
            other_boundary_pts = boundary_pts[~current_boundary_pts_mask]

            current_boundary_fit_dists2, _ = mash_cpp.toChamferDistance(
                current_boundary_pts.unsqueeze(0).type(gt_points.dtype),
                other_boundary_pts.unsqueeze(0).type(gt_points.dtype),
            )[:2]

            current_boundary_connect_dist = torch.sqrt(
                current_boundary_fit_dists2 + EPSILON
            )

            current_boundary_connect_loss = torch.mean(current_boundary_connect_dist)

            boundary_connect_loss = (
                boundary_connect_loss + current_boundary_connect_loss
            )

        boundary_connect_loss = boundary_connect_loss / self.mash.anchor_num

        manifold_loss_weight = self.epoch / (self.train_epoch - 1)

        fit_loss_weight = 1.0
        coverage_loss_weight = 0.25 + 0.75 * manifold_loss_weight
        boundary_connect_loss_weight = 0.01 + 0.99 * manifold_loss_weight

        weighted_fit_loss = fit_loss_weight * fit_loss
        weighted_coverage_loss = coverage_loss_weight * coverage_loss
        weighted_boundary_connect_loss = (
            boundary_connect_loss_weight * boundary_connect_loss
        )

        loss = (
            weighted_fit_loss + weighted_coverage_loss + weighted_boundary_connect_loss
        )

        loss.backward()

        self.optimizer.step()

        loss_dict = {
            "State/boundary_pts": boundary_pts.shape[0],
            "State/inner_pts": inner_pts.shape[0],
            "Train/epoch": self.epoch,
            "Train/weighted_fit_loss": toNumpy(weighted_fit_loss),
            "Train/weighted_coverage_loss": toNumpy(weighted_coverage_loss),
            "Train/weighted_boundary_connect_loss": toNumpy(
                weighted_boundary_connect_loss
            ),
            "Train/loss": toNumpy(loss),
            "Metric/chamfer_distance": toNumpy(fit_loss) + toNumpy(coverage_loss),
        }

        return loss_dict

    def fitMash(
        self,
        gt_points: torch.Tensor,
    ) -> bool:
        self.mash.setGradState(True)

        self.updateLr(self.fit_lr)

        print("[INFO][MashModelOp::fitMash]")
        print("\t start warm up mash ...")
        pbar = tqdm(total=self.fit_step_num)
        while self.step < self.fit_step_num:
            if self.render:
                if self.step % self.render_freq == 0:
                    self.renderMash(gt_points)

            loss_dict = self.trainStep(
                gt_points,
            )

            assert isinstance(loss_dict, dict)
            if self.logger.isValid():
                for key, item in loss_dict.items():
                    self.logger.addScalar(key, item, self.step)

            loss = loss_dict["Train/loss"]

            pbar.set_description("LOSS %.6f" % (loss,))

            self.autoSaveMash("train")

            self.step += 1
            pbar.update(1)
        return True

    def warmUpMash(
        self,
        gt_points: torch.Tensor,
    ) -> bool:
        self.mash.setGradState(True)

        print("[INFO][MashModelOp::warmUpMash]")
        print("\t start warm up mash ...")
        pbar = tqdm(total=self.warm_step_num)
        while self.step < self.warm_step_num:
            if self.render:
                if self.step % self.render_freq == 0:
                    self.renderMash(gt_points)

            current_lr = 1.0 * (self.step + 1) / self.warm_step_num * self.lr

            self.updateLr(current_lr)

            loss_dict = self.trainStep(
                gt_points,
            )

            assert isinstance(loss_dict, dict)
            if self.logger.isValid():
                for key, item in loss_dict.items():
                    self.logger.addScalar(key, item, self.step)

            loss = loss_dict["Train/loss"]

            pbar.set_description("LOSS %.6f" % (loss,))

            self.autoSaveMash("train")

            self.step += 1
            pbar.update(1)
        return True

    def trainMash(
        self,
        gt_points: torch.Tensor,
    ) -> bool:
        self.mash.setGradState(True)

        self.updateLr(self.lr)

        min_loss = float("inf")
        min_loss_reached_time = 0

        print("[INFO][MashModelOp::trainMash]")
        print(
            "\t start train mash epoch",
            self.epoch + 1,
            "/",
            self.train_epoch,
            "...",
        )
        pbar = tqdm()
        while True:
            if self.render:
                if self.step % self.render_freq == 0:
                    self.renderMash(gt_points)

            loss_dict = self.trainStep(
                gt_points,
            )

            assert isinstance(loss_dict, dict)
            if self.logger.isValid():
                for key, item in loss_dict.items():
                    self.logger.addScalar(key, item, self.step)

            loss = loss_dict["Train/loss"]

            pbar.set_description("LOSS %.6f" % (loss,))

            self.autoSaveMash("train")

            self.step += 1
            pbar.update(1)

            if loss < min_loss:
                min_loss = loss
                min_loss_reached_time = 0
            else:
                min_loss_reached_time += 1

            self.logger.addScalar(
                "Train/patience", self.patience - min_loss_reached_time, self.step
            )

            if min_loss_reached_time >= self.patience:
                break

        self.epoch += 1
        return True

    def autoTrainMash(
        self,
        gt_points_num: int = 400000,
    ) -> bool:
        print("[INFO][Trainer::autoTrainMash]")
        print("\t start auto train Mash...")
        print(
            "\t degree: mask:",
            self.mash.mask_degree_max,
            "sh:",
            self.mash.sh_degree_max,
        )

        if self.mash.device == "cpu":
            gt_points_dtype = self.mash.dtype
        else:
            gt_points_dtype = torch.float32

        if self.gt_points is None:
            if not self.mesh.isValid():
                print("[ERROR][Trainer::autoTrainMash]")
                print("\t mesh is not valid!")
                return False

            self.gt_points = self.mesh.toSamplePoints(gt_points_num)

        gt_points = (
            torch.from_numpy(self.gt_points)
            .type(gt_points_dtype)
            .to(self.mash.device)
            .reshape(1, -1, 3)
        )

        self.fitMash(gt_points)

        for _ in range(self.train_epoch):
            self.trainMash(gt_points)

        """
            if self.upperSHDegree():
                print("[INFO][Trainer::autoTrainMash]")
                print("\t upperSHDegree success!")
                print("\t start auto train Mash...")
                print(
                    "\t degree: mask:",
                    self.mash.mask_degree_max,
                    "sh:",
                    self.mash.sh_degree_max,
                )
                continue

            if self.upperMaskDegree():
                print("[INFO][Trainer::autoTrainMash]")
                print("\t upperMaskDegree success!")
                print("\t start auto train Mash...")
                print(
                    "\t degree: mask:",
                    self.mash.mask_degree_max,
                    "sh:",
                    self.mash.sh_degree_max,
                )
                continue

            break
        """

        return True

    def autoSaveMash(self, state_info: str) -> bool:
        if self.save_result_folder_path is None:
            return False

        save_file_path = (
            self.save_result_folder_path
            + str(self.save_file_idx)
            + "_"
            + state_info
            + ".npy"
        )

        save_mash = deepcopy(self.mash)
        save_mash.saveParamsFile(save_file_path, True)

        self.save_file_idx += 1
        return True

    def renderMash(self, gt_points: torch.Tensor) -> bool:
        if self.o3d_viewer is None:
            print("[ERROR][Trainer::renderMash]")
            print("\t o3d_viewer is None!")
            return False

        with torch.no_grad():
            self.o3d_viewer.clearGeometries()

            mesh_abb_length = 2.0 * self.mesh.toABBLength()
            if mesh_abb_length == 0:
                mesh_abb_length = 1.1

            gt_pcd = getPointCloud(toNumpy(gt_points.squeeze(0)))
            gt_pcd.translate([-mesh_abb_length, 0, 0])
            self.o3d_viewer.addGeometry(gt_pcd)

            detect_points = toNumpy(self.mash.toSamplePoints())
            pcd = getPointCloud(detect_points)
            self.o3d_viewer.addGeometry(pcd)

            # self.mesh.paintJetColorsByPoints(detect_points)
            mesh = self.mesh.toO3DMesh()
            mesh.translate([mesh_abb_length, 0, 0])
            self.o3d_viewer.addGeometry(mesh)

            """
            for j in range(self.mash.mask_params.shape[0]):
                view_cone = self.toO3DViewCone(j)
                view_cone.translate([-mesh_abb_length, 0, 0])
                self.o3d_viewer.addGeometry(view_cone)

                # inv_sphere = self.toO3DInvSphere(j)
                # inv_sphere.translate([-30, 0, 0])
                # self.o3d_viewer.addGeometry(inv_sphere)
            """

            self.o3d_viewer.update()

            if self.render_init_only:
                self.o3d_viewer.run()
                exit()
        return True
