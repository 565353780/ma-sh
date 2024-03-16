import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Union
from torch.optim import AdamW as OPTIMIZER
from torch.optim.lr_scheduler import ReduceLROnPlateau as SCHEDULER

from ma_sh.Config.constant import EPSILON
from ma_sh.Config.degree import MAX_MASK_DEGREE, MAX_SH_DEGREE
from ma_sh.Data.mesh import Mesh
from ma_sh.Loss.chamfer_distance import chamferDistance
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Method.time import getCurrentTime
from ma_sh.Method.path import createFileFolder, removeFile, renameFile
from ma_sh.Model.mash import Mash
from ma_sh.Module.logger import Logger
from ma_sh.Module.o3d_viewer import O3DViewer


class Trainer(object):
    def __init__(
        self,
        anchor_num: int = 4,
        mask_degree_max: int = 5,
        sh_degree_max: int = 3,
        mask_boundary_sample_num: int = 10,
        sample_polar_num: int = 10,
        idx_dtype=torch.int64,
        dtype=torch.float64,
        device: str = "cpu",
        epoch: int = 1000,
        lr: float = 1e-2,
        weight_decay: float = 1e-10,
        factor: float = 0.8,
        patience: int = 10,
        min_lr: float = 1e-3,
        render: bool = False,
        save_folder_path: Union[str, None] = None,
        direction_upscale: int = 4,
    ) -> None:
        self.mash = Mash(
            anchor_num,
            mask_degree_max,
            sh_degree_max,
            mask_boundary_sample_num,
            sample_polar_num,
            idx_dtype,
            dtype,
            device,
        )

        self.use_inv = False

        self.epoch = epoch

        self.step = 0
        self.loss_min = float("inf")

        self.best_params_dict = {}

        self.lr = lr
        self.weight_decay = weight_decay
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr

        self.render = render

        self.save_folder_path = None
        self.save_file_idx = 0
        self.logger = Logger()

        # TODO: can start from 0 and auto upperDegrees later

        self.mesh = Mesh()

        self.loadRecords(save_folder_path)

        self.direction_upscale = direction_upscale
        self.fps_scale = 1.0 / self.direction_upscale

        if self.render:
            self.o3d_viewer = O3DViewer()
            self.o3d_viewer.createWindow()
        return

    def loadRecords(self, save_folder_path: Union[str, None] = None) -> bool:
        self.save_file_idx = 0

        current_time = getCurrentTime()

        if save_folder_path is None:
            self.save_folder_path = "./output/" + current_time + "/"
            log_folder_path = "./logs/" + current_time + "/"
        else:
            self.save_folder_path = save_folder_path
            log_folder_path = save_folder_path + "../logs/" + current_time + "/"

        os.makedirs(self.save_folder_path, exist_ok=True)
        os.makedirs(log_folder_path, exist_ok=True)
        self.logger.setLogFolder(log_folder_path)
        return True

    def loadMeshFile(self, mesh_file_path: str) -> bool:
        if not os.path.exists(mesh_file_path):
            print("[ERROR][Trainer::loadMeshFile]")
            print("\t mesh file not exist!")
            print("\t mesh_file_path:", mesh_file_path)
            return False

        return self.mesh.loadMesh(mesh_file_path)

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
        }
        return True

    def loadParams(
        self,
        mask_params: torch.Tensor,
        sh_params: torch.Tensor,
        rotate_vectors: torch.Tensor,
        positions: torch.Tensor,
    ) -> bool:
        self.mash.loadParams(mask_params, sh_params, rotate_vectors, positions)

        self.updateBestParams()
        return True

    def loadParamsNp(
        self,
        mask_params_np: np.ndarray,
        sh_params_np: np.ndarray,
        rotate_vectors_np: np.ndarray,
        positions_np: np.ndarray,
    ) -> bool:
        mask_params = (
            torch.from_numpy(mask_params_np).type(self.mash.dtype).to(self.mash.device)
        )
        sh_params = (
            torch.from_numpy(sh_params_np).type(self.mash.dtype).to(self.mash.device)
        )
        rotate_vectors = (
            torch.from_numpy(rotate_vectors_np)
            .type(self.mash.dtype)
            .to(self.mash.device)
        )
        positions = (
            torch.from_numpy(positions_np).type(self.mash.dtype).to(self.mash.device)
        )

        return self.loadParams(mask_params, sh_params, rotate_vectors, positions)

    def loadBestParams(
        self,
        mask_params: torch.Tensor,
        sh_params: torch.Tensor,
        rotate_vectors: torch.Tensor,
        positions: torch.Tensor,
    ) -> bool:
        self.best_params_dict["mask_params"] = mask_params
        self.best_params_dict["sh_params"] = sh_params
        self.best_params_dict["rotate_vectors"] = rotate_vectors
        self.best_params_dict["positions"] = positions
        return True

    def loadBestParamsNp(
        self,
        mask_params_np: np.ndarray,
        sh_params_np: np.ndarray,
        rotate_vectors_np: np.ndarray,
        positions_np: np.ndarray,
    ) -> bool:
        mask_params = (
            torch.from_numpy(mask_params_np).type(self.mash.dtype).to(self.mash.device)
        )
        sh_params = (
            torch.from_numpy(sh_params_np).type(self.mash.dtype).to(self.mash.device)
        )
        rotate_vectors = (
            torch.from_numpy(rotate_vectors_np)
            .type(self.mash.dtype)
            .to(self.mash.device)
        )
        positions = (
            torch.from_numpy(positions_np).type(self.mash.dtype).to(self.mash.device)
        )

        return self.loadBestParams(mask_params, sh_params, rotate_vectors, positions)

    def setBestParams(self) -> bool:
        self.mash.loadParams(
            self.best_params_dict["mask_params"],
            self.best_params_dict["sh_params"],
            self.best_params_dict["rotate_vectors"],
            self.best_params_dict["positions"],
        )
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

    def trainStep(
        self,
        optimizer,
        gt_points: torch.Tensor,
    ) -> Union[dict, None]:
        optimizer.zero_grad()

        detect_points = self.mash.toSamplePoints()

        fit_dists2, coverage_dists2 = chamferDistance(
            detect_points.reshape(1, -1, 3).type(gt_points.dtype),
            gt_points,
            self.mash.device == "cpu",
        )[:2]

        fit_dists = torch.mean(torch.sqrt(fit_dists2) + EPSILON)
        coverage_dists = torch.mean(torch.sqrt(coverage_dists2) + EPSILON)

        fit_loss = torch.mean(fit_dists)
        coverage_loss = torch.mean(coverage_dists)

        fit_safe_scale = 0.2
        current_lr_coeff = np.log(self.getLr(optimizer))
        min_lr_coeff = np.log(self.min_lr)
        lr_coeff = np.log(self.lr)
        lr_remain_scale = (current_lr_coeff - min_lr_coeff) / (lr_coeff - min_lr_coeff)
        fit_scale = fit_safe_scale + (1.0 - fit_safe_scale) * (1.0 - lr_remain_scale)
        coverage_scale = fit_safe_scale + (1.0 - fit_safe_scale) * lr_remain_scale
        fit_loss = fit_scale * torch.mean(fit_dists)
        coverage_loss = coverage_scale * torch.mean(coverage_dists)

        loss = fit_loss + coverage_loss

        loss.backward()

        """
        nn.utils.clip_grad_norm_(self.mash.mask_params, max_norm=1e5, norm_type=2)
        nn.utils.clip_grad_norm_(self.mash.sh_params, max_norm=1e5, norm_type=2)
        nn.utils.clip_grad_norm_(self.mash.rotate_vectors, max_norm=1e5, norm_type=2)
        nn.utils.clip_grad_norm_(self.mash.positions, max_norm=1e5, norm_type=2)

        if torch.isnan(self.mash.mask_params.grad).any():
            print("grad contains nan, set it to 0!")
            self.mash.mask_params.grad[torch.isnan(self.mash.mask_params.grad)] = 0.0
            exit()
        if torch.isnan(self.mash.sh_params.grad).any():
            print("grad contains nan, set it to 0!")
            self.mash.sh_params.grad[torch.isnan(self.mash.sh_params.grad)] = 0.0
            exit()
        if torch.isnan(self.mash.rotate_vectors.grad).any():
            print("grad contains nan, set it to 0!")
            self.mash.rotate_vectors.grad[
                torch.isnan(self.mash.rotate_vectors.grad)
            ] = 0.0
            exit()
        if torch.isnan(self.mash.positions.grad).any():
            print("grad contains nan, set it to 0!")
            self.mash.positions.grad[torch.isnan(self.mash.positions.grad)] = 0.0
            exit()
        """

        optimizer.step()

        loss_dict = {
            "fit_loss": fit_loss.detach().clone().cpu().numpy(),
            "coverage_loss": coverage_loss.detach().clone().cpu().numpy(),
            "chamfer_distance": (torch.mean(fit_dists) + torch.mean(coverage_dists))
            .detach()
            .clone()
            .cpu()
            .numpy(),
            "loss": loss.detach().clone().cpu().numpy(),
        }

        return loss_dict

    def trainMash(
        self,
        gt_points_num: int = 10000,
    ) -> bool:
        self.mash.setGradState(True)

        optimizer = OPTIMIZER(
            [
                self.mash.mask_params,
                self.mash.sh_params,
                self.mash.rotate_vectors,
                self.mash.positions,
            ],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = SCHEDULER(
            optimizer,
            mode="min",
            factor=self.factor,
            patience=self.patience,
            min_lr=self.min_lr,
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

        best_result_reached_num = 0

        print("[INFO][MashModelOp::train]")
        print("\t start training ...")
        pbar = tqdm(total=self.epoch)
        pbar.update(self.step)
        while self.step < self.epoch:
            if self.render and self.step % 5 == 0:
                with torch.no_grad():
                    self.o3d_viewer.clearGeometries()

                    detect_points = (
                        self.mash.toSamplePoints()
                        .detach()
                        .clone()
                        .cpu()
                        .numpy()
                        .reshape(-1, 3)
                    )
                    pcd = getPointCloud(detect_points)
                    self.o3d_viewer.addGeometry(pcd)

                    mesh_abb_length = 2.0 * self.mesh.toABBLength()

                    self.mesh.paintJetColorsByPoints(detect_points)
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

                    if False:
                        self.o3d_viewer.run()
                        exit()

            loss_dict = self.trainStep(
                optimizer,
                gt_points,
            )

            assert isinstance(loss_dict, dict)
            for key, item in loss_dict.items():
                self.logger.addScalar("Train/" + key, item, self.step)
            self.logger.addScalar("Train/lr", self.getLr(optimizer), self.step)

            scheduler.step(loss_dict["loss"])

            pbar.set_description(
                "LOSS %.6f LR %.4f" % (loss_dict["loss"], self.getLr(optimizer))
            )

            self.updateBestParams(loss_dict["loss"])

            if self.getLr(optimizer) <= self.min_lr:
                best_result_reached_num += 1

            if best_result_reached_num > self.patience:
                break

            self.autoSaveMash("train")

            self.step += 1
            pbar.update(1)

        return True

    def autoTrainMash(
        self,
        gt_points_num: int = 10000,
    ) -> bool:
        # TODO: need to adaptive upper degrees here for more stable optimizing
        # self.sh_3d_degree_max = self.max_sh_3d_degree
        # self.sh_2d_degree_max = self.max_sh_2d_degree

        self.mash.reset()

        print("[INFO][Trainer::autoTrainMash]")
        print("\t start auto train Mash...")
        print(
            "\t degree: mask:",
            self.mash.mask_degree_max,
            "sh:",
            self.mash.sh_degree_max,
        )

        while True:
            while not self.trainMash(gt_points_num):
                self.mash.reset()
                continue

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

    def saveParams(self, save_params_file_path: str, overwrite: bool = False) -> bool:
        if os.path.exists(save_params_file_path):
            if overwrite:
                removeFile(save_params_file_path)
            else:
                print("[WARN][Trainer::saveParams]")
                print("\t save params file already exist!")
                print("\t save_params_file_path:", save_params_file_path)
                return False

        params_dict = {
            "mask_degree_max": self.mash.mask_degree_max,
            "sh_degree_max": self.mash.sh_degree_max,
            "use_inv": self.use_inv,
            "mask_params": self.mash.mask_params.detach().clone().cpu().numpy(),
            "sh_params": self.mash.sh_params.detach().clone().cpu().numpy(),
            "rotate_vectors": self.mash.rotate_vectors.detach().clone().cpu().numpy(),
            "positions": self.mash.positions.detach().clone().cpu().numpy(),
        }

        for key, item in self.best_params_dict.items():
            params_dict["best_" + key] = item.detach().clone().cpu().numpy()

        createFileFolder(save_params_file_path)

        tmp_save_params_file_path = save_params_file_path.split(".npy")[0] + "_tmp.npy"
        removeFile(tmp_save_params_file_path)

        np.save(tmp_save_params_file_path, params_dict)
        renameFile(tmp_save_params_file_path, save_params_file_path)
        return True

    def loadParamsFile(self, params_file_path: str) -> bool:
        if not os.path.exists(params_file_path):
            print("[ERROR][Trainer::loadParamsFile]")
            print("\t params file not exist!")
            print("\t params_file_path:", params_file_path)
            return False

        params_dict = np.load(params_file_path, allow_pickle=True).item()

        self.mash.mask_degree_max = int(params_dict["mask_degree_max"])
        self.mash.sh_degree_max = int(params_dict["sh_degree_max"])
        self.use_inv = bool(params_dict["use_inv"])

        self.loadParamsNp(
            params_dict["mask_params"],
            params_dict["sh_params"],
            params_dict["rotate_vectors"],
            params_dict["positions"],
        )
        self.loadBestParamsNp(
            params_dict["best_mask_params"],
            params_dict["best_sh_params"],
            params_dict["best_rotate_vectors"],
            params_dict["best_positions"],
        )
        return True

    def autoSaveMash(self, state_info: str) -> bool:
        save_file_path = (
            self.save_folder_path + str(self.save_file_idx) + "_" + state_info + ".npy"
        )

        self.saveParams(save_file_path, True)

        self.save_file_idx += 1
        return True
