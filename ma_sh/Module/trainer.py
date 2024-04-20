import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Union
from copy import deepcopy
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import (
    LRScheduler,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)

from ma_sh.Config.constant import EPSILON
from ma_sh.Config.weights import W0
from ma_sh.Config.degree import MAX_MASK_DEGREE, MAX_SH_DEGREE
from ma_sh.Data.mesh import Mesh
from ma_sh.Loss.chamfer_distance import chamferDistance
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Method.time import getCurrentTime
from ma_sh.Model.mash import Mash
from ma_sh.Module.logger import Logger
from ma_sh.Module.o3d_viewer import O3DViewer


class Trainer(object):
    def __init__(
        self,
        anchor_num: int = 40,
        mask_degree_max: int = 4,
        sh_degree_max: int = 4,
        mask_boundary_sample_num: int = 18,
        sample_polar_num: int = 4000,
        sample_point_scale: float = 0.4,
        use_inv: bool = True,
        idx_dtype=torch.int64,
        dtype=torch.float64,
        device: str = "cpu",
        warm_epoch_step_num: int = 20,
        warm_epoch_num: int = 10,
        finetune_step_num: int = 400,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        factor: float = 0.9,
        patience: int = 1,
        min_lr: float = 1e-4,
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

        self.warm_epoch_step_num = warm_epoch_step_num
        self.warm_epoch_num = warm_epoch_num

        self.finetune_step_num = finetune_step_num

        self.step = 0
        self.loss_min = float("inf")

        self.best_params_dict = {}

        self.lr = lr
        self.weight_decay = weight_decay
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr

        self.render = render
        self.render_freq = render_freq
        self.render_init_only = render_init_only

        self.save_result_folder_path = save_result_folder_path
        self.save_log_folder_path = save_log_folder_path
        self.save_file_idx = 0
        self.logger = Logger()

        self.mesh = Mesh()

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

        surface_dist = 0.1

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

    def updateBestParams(self, loss: Union[float, None] = None) -> bool:
        if loss is not None:
            if loss >= self.loss_min:
                return False

            self.loss_min = loss

        self.best_params_dict = {
            "mask_params": self.mash.mask_params.detach().clone(),
            "sh_params": self.mash.sh_params.detach().clone(),
            "rotate_vectors": self.mash.rotate_vectors.detach().clone(),
            "positions": self.mash.positions.detach().clone(),
            "use_inv": self.mash.use_inv,
        }
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

        self.updateBestParams()
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

        self.updateBestParams()
        return True

    def upperSHDegree(self) -> bool:
        if self.mash.sh_degree_max == MAX_SH_DEGREE:
            return False

        if not self.mash.updateSHDegree(self.mash.sh_degree_max + 1):
            print("[ERROR][Trainer::upperMaskDegree]")
            print("\t updateSHDegree failed!")
            return False

        self.updateBestParams()
        return True

    def getLr(self, optimizer) -> float:
        return optimizer.state_dict()["param_groups"][0]["lr"]

    def toTrainStepNum(self, scheduler: LRScheduler) -> int:
        if not isinstance(scheduler, CosineAnnealingWarmRestarts):
            return self.finetune_step_num

        if scheduler.T_mult == 1:
            warm_epoch_num = scheduler.T_0 * self.warm_epoch_num
        else:
            warm_epoch_num = int(
                scheduler.T_mult
                * (1.0 - pow(scheduler.T_mult, self.warm_epoch_num))
                / (1.0 - scheduler.T_mult)
            )

        return self.warm_epoch_step_num * warm_epoch_num

    def trainStep(
        self,
        optimizer: Optimizer,
        gt_points: torch.Tensor,
    ) -> Union[dict, None]:
        optimizer.zero_grad()

        detect_points = self.mash.toSamplePoints()

        fit_dists2, coverage_dists2 = chamferDistance(
            detect_points.reshape(1, -1, 3).type(gt_points.dtype),
            gt_points,
            "cuda" not in self.mash.device,
        )[:2]

        fit_dists = torch.sqrt(fit_dists2 + EPSILON)
        coverage_dists = torch.sqrt(coverage_dists2 + EPSILON)

        fit_loss = torch.mean(fit_dists)
        coverage_loss = torch.mean(coverage_dists)

        loss = fit_loss + coverage_loss

        loss.backward()

        optimizer.step()

        loss_dict = {
            "fit_loss": fit_loss.detach().clone().cpu().numpy(),
            "coverage_loss": coverage_loss.detach().clone().cpu().numpy(),
            "loss": loss.detach().clone().cpu().numpy(),
        }

        return loss_dict

    def checkStop(
        self, optimizer: Optimizer, scheduler: LRScheduler, loss_dict: dict
    ) -> bool:
        if not isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step(loss_dict["loss"])

            if self.getLr(optimizer) == self.min_lr:
                self.min_lr_reach_time += 1

            return self.min_lr_reach_time > self.patience

        current_warm_epoch = self.step / self.warm_epoch_step_num
        scheduler.step(current_warm_epoch)

        return current_warm_epoch >= self.warm_epoch_num

    def trainMash(
        self,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        gt_points: torch.Tensor,
    ) -> bool:
        self.mash.setGradState(True)

        train_step_num = self.toTrainStepNum(scheduler)
        final_step = self.step + train_step_num

        print("[INFO][MashModelOp::train]")
        print("\t start training ...")
        pbar = tqdm(total=final_step)
        pbar.update(self.step)
        while self.step < final_step:
            if self.render and self.step % self.render_freq == 0:
                assert self.o3d_viewer is not None
                with torch.no_grad():
                    self.o3d_viewer.clearGeometries()

                    mesh_abb_length = 2.0 * self.mesh.toABBLength()

                    gt_pcd = getPointCloud(gt_points.reshape(-1, 3).cpu().numpy())
                    gt_pcd.translate([-mesh_abb_length, 0, 0])
                    self.o3d_viewer.addGeometry(gt_pcd)

                    detect_points = (
                        self.mash.toSamplePoints().detach().clone().cpu().numpy()
                    )
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

            loss_dict = self.trainStep(
                optimizer,
                gt_points,
            )

            assert isinstance(loss_dict, dict)
            if self.logger.isValid():
                for key, item in loss_dict.items():
                    self.logger.addScalar("Train/" + key, item, self.step)
                self.logger.addScalar("Train/lr", self.getLr(optimizer), self.step)

            self.updateBestParams(loss_dict["loss"])

            pbar.set_description(
                "LOSS %.6f LR %.4f"
                % (
                    loss_dict["loss"],
                    self.getLr(optimizer) / self.lr,
                )
            )

            self.autoSaveMash("train")

            if self.checkStop(optimizer, scheduler, loss_dict):
                break

            self.step += 1
            pbar.update(1)

        return True

    def autoTrainMash(
        self,
        gt_points_num: int = 10000,
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
        gt_points = (
            torch.from_numpy(self.mesh.toSamplePoints(gt_points_num))
            .type(gt_points_dtype)
            .to(self.mash.device)
            .reshape(1, -1, 3)
        )

        self.gt_points_ = gt_points

        while True:
            optimizer = AdamW(
                [
                    self.mash.mask_params,
                    self.mash.sh_params,
                    self.mash.rotate_vectors,
                    self.mash.positions,
                ],
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            warm_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=1)
            finetune_scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.factor,
                patience=self.patience,
                min_lr=self.min_lr,
            )

            self.trainMash(optimizer, warm_scheduler, gt_points)
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.lr
            self.trainMash(optimizer, finetune_scheduler, gt_points)

            break

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
        save_mash.loadParamsDict(self.best_params_dict)
        save_mash.saveParamsFile(save_file_path, True)

        self.save_file_idx += 1
        return True
