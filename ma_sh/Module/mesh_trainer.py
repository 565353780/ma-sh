import os
import torch
from time import time
from typing import Union
from torch.optim import AdamW

from mesh_graph_cut.Module.mesh_graph_cutter import MeshGraphCutter

from ma_sh.Config.constant import EPSILON
from ma_sh.Method.data import toNumpy
from ma_sh.Model.simple_mash import SimpleMash
from ma_sh.Module.base_trainer import BaseTrainer


class MeshTrainer(BaseTrainer):
    def __init__(
        self,
        anchor_num: int = 400,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        sample_phi_num: int = 40,
        sample_theta_num: int = 40,
        use_inv: bool = True,
        idx_dtype=torch.int64,
        dtype=torch.float32,
        device: str = "cuda",
        lr: float = 2e-3,
        min_lr: float = 1e-3,
        warmup_step_num: int = 80,
        warmup_epoch: int = 4,
        factor: float = 0.8,
        patience: int = 2,
        render: bool = False,
        render_freq: int = 1,
        render_init_only: bool = False,
        save_freq: int = 1,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
    ) -> None:
        self.warmup_step_num = warmup_step_num
        self.warmup_epoch = warmup_epoch
        self.factor = factor

        self.mash = SimpleMash(
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

        self.optimizer = AdamW(
            [
                self.mash.mask_params,
                self.mash.sh_params,
                self.mash.rotate_vectors,
                self.mash.positions,
            ],
            lr=lr,
        )

        super().__init__(
            lr,
            min_lr,
            patience,
            render,
            render_freq,
            render_init_only,
            save_freq,
            save_result_folder_path,
            save_log_folder_path,
        )
        return

    def loadMeshFile(
        self,
        mesh_file_path: str,
        points_per_submesh: int = 8192,
    ) -> bool:
        if not os.path.exists(mesh_file_path):
            print("[ERROR][MeshTrainer::loadMeshFile]")
            print("\t mesh file not exist!")
            print("\t mesh_file_path:", mesh_file_path)
            return False

        if not self.mesh.loadMesh(mesh_file_path):
            print("[ERROR][BaseTrainer::loadMeshFile]")
            print("\t loadMesh failed!")
            print("\t mesh_file_path:", mesh_file_path)
            return False

        mesh_graph_cutter = MeshGraphCutter(mesh_file_path)
        mesh_graph_cutter.cutMesh(self.mash.anchor_num, points_per_submesh)

        self.gt_points = mesh_graph_cutter.sub_mesh_sample_points

        fps_positions = mesh_graph_cutter.vertices[mesh_graph_cutter.fps_vertex_idxs]

        fps_normals = mesh_graph_cutter.vertex_normals[
            mesh_graph_cutter.fps_vertex_idxs
        ]

        sh_params = torch.zeros_like(self.mash.sh_params)
        sh_params[:, 0] = self.surface_dist / W0[0]

        self.mash.loadParams(
            sh_params=sh_params,
            positions=fps_positions + self.surface_dist * fps_normals,
            face_forward_vectors=-fps_normals,
        )
        return True

    def trainStep(
        self,
        gt_points: torch.Tensor,
        fit_loss_weight: float,
        coverage_loss_weight: float,
        boundary_connect_loss_weight: float,
    ) -> Union[dict, None]:
        self.optimizer.zero_grad()

        start = time()
        # boundary_pts, inner_pts = self.mash.toSamplePoints()[:2]
        mash_pts = self.mash.toSimpleSamplePoints().reshape(self.mash.anchor_num, -1, 3)
        self.sample_mash_time += time() - start

        start = time()
        fit_loss = torch.tensor(0.0).type(gt_points.dtype).to(gt_points.device)
        coverage_loss = torch.zeros_like(fit_loss)

        fit_dists2, coverage_dists2 = ChamferDistances.namedAlgo("triton")(
            mash_pts, gt_points
        )[:2]

        fit_dists = torch.sqrt(fit_dists2 + EPSILON)
        coverage_dists = torch.sqrt(coverage_dists2 + EPSILON)

        fit_loss = torch.mean(fit_dists)
        coverage_loss = torch.mean(coverage_dists)

        spend = time() - start
        self.fit_loss_time += spend / 2.0
        self.coverage_loss_time += spend / 2.0

        weighted_fit_loss = fit_loss_weight * fit_loss
        weighted_coverage_loss = coverage_loss_weight * coverage_loss

        loss = weighted_fit_loss + weighted_coverage_loss

        if torch.isnan(loss).any():
            print("[ERROR][BaseTrainer::trainStep]")
            print("\t loss is nan!")
            print("\t\t fit_loss:", fit_loss)
            print("\t\t coverage_loss:", coverage_loss)
            return None

        loss.backward()
        torch.cuda.empty_cache()  # 清空反向传播后不再需要的GPU缓存

        self.mash.clearNanGrads()

        self.optimizer.step()
        torch.cuda.empty_cache()  # 清空优化器步骤后不再需要的GPU缓存

        chamfer_distance = toNumpy(fit_loss) + toNumpy(coverage_loss)
        self.error = chamfer_distance

        loss_dict = {
            "State/boundary_pts": boundary_pts.shape[0],
            "State/inner_pts": inner_pts.shape[0],
            "Train/epoch": self.epoch,
            "Train/fit_loss": toNumpy(fit_loss),
            "Train/coverage_loss": toNumpy(coverage_loss),
            "Train/weighted_fit_loss": toNumpy(weighted_fit_loss),
            "Train/weighted_coverage_loss": toNumpy(weighted_coverage_loss),
            "Train/loss": toNumpy(loss),
            "Metric/chamfer_distance": chamfer_distance,
        }

        return loss_dict

    def autoTrainMash(
        self,
        gt_points_num: int = 400000,
    ) -> bool:
        print("[INFO][MeshTrainer::autoTrainMash]")
        print("\t start auto train SimpleMash...")
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

        gt_points = torch.from_numpy(self.gt_points).to(
            self.mash.device, dtype=gt_points_dtype
        )

        print("[INFO][MeshTrainer::autoTrainMash]")
        print("\t start warmUpEpoch...")
        if not self.warmUpEpoch(
            self.lr, gt_points, 1.0, 0.5, 0.0, self.warmup_step_num
        ):
            print("[ERROR][Trainer::autoTrainMash]")
            print("\t warmUpEpoch failed!")
            return False

        print("[INFO][Trainer::autoTrainMash]")
        print("\t start trainEpoch with adaptive loss...")
        for i in range(self.warmup_epoch):
            fit_loss_weight = 1.0

            manifold_loss_weight = i / (self.warmup_epoch - 1)

            coverage_loss_weight = 0.5 + 0.5 * manifold_loss_weight

            if not self.trainEpoch(
                self.lr, gt_points, fit_loss_weight, coverage_loss_weight, 0
            ):
                print("[ERROR][Trainer::autoTrainMash]")
                print("\t trainEpoch failed!")
                return False

        print("[INFO][Trainer::autoTrainMash]")
        print("\t start trainEpoch with adaptive lr...")
        current_lr = self.lr
        current_finetune_epoch = 1

        while current_lr > self.min_lr:
            current_lr *= self.factor
            if current_lr < self.min_lr:
                current_lr = self.min_lr

            current_finetune_epoch += 1

            fit_loss_weight = 1.0
            coverage_loss_weight = 1.0

            manifold_loss_weight = 1.0 / current_finetune_epoch

            self.trainEpoch(
                current_lr, gt_points, fit_loss_weight, coverage_loss_weight, 0
            )

            if current_lr == self.min_lr:
                break

        self.autoSavePcd("final", add_idx=False)
        self.autoSaveMash("final")

        total_time = time() - self.start_time

        print("[INFO][Trainer::autoTrainMash]")
        print("\t training finished! metrics:")
        print("\t surface sampling:", self.sample_mash_time)
        print("\t fit loss:", self.fit_loss_time)
        print("\t coverage loss:", self.coverage_loss_time)
        print("\t total:", total_time)
        print("\t error:", self.error)

        if self.o3d_viewer is not None:
            self.o3d_viewer.run()

        return True
