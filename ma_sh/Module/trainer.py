import torch
from typing import Union
from torch.optim import AdamW

from ma_sh.Model.mash import Mash
from ma_sh.Model.simple_mash import SimpleMash
from ma_sh.Module.base_trainer import BaseTrainer

mode = 'mash'

class Trainer(BaseTrainer):
    def __init__(
        self,
        anchor_num: int = 400,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        mask_boundary_sample_num: int = 90,
        sample_polar_num: int = 1000,
        sample_point_scale: float = 0.8,
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

        if mode == 'mash':
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
        elif mode == 'simple_mash':
            self.mash = SimpleMash(
                anchor_num,
                mask_degree_max,
                sh_degree_max,
                10,
                10,
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

    def autoTrainMash(
        self,
        gt_points_num: int = 400000,
    ) -> bool:
        boundary_connect_loss_weight_max = 0.1

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

        print("[INFO][Trainer::autoTrainMash]")
        print("\t start warmUpEpoch...")
        if not self.warmUpEpoch(self.lr, gt_points, 1.0, 0.5, 0.0, self.warmup_step_num):
            print('[ERROR][Trainer::autoTrainMash]')
            print('\t warmUpEpoch failed!')
            return False

        print("[INFO][Trainer::autoTrainMash]")
        print("\t start trainEpoch with adaptive loss...")
        for i in range(self.warmup_epoch):
            fit_loss_weight = 1.0

            manifold_loss_weight = i / (self.warmup_epoch - 1)

            coverage_loss_weight = 0.5 + 0.5 * manifold_loss_weight
            boundary_connect_loss_weight = (
                0.1 + 0.9 * manifold_loss_weight
            ) * boundary_connect_loss_weight_max

            if not self.trainEpoch(
                self.lr,
                gt_points,
                fit_loss_weight,
                coverage_loss_weight,
                boundary_connect_loss_weight,
            ):
                print('[ERROR][Trainer::autoTrainMash]')
                print('\t trainEpoch failed!')
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

            boundary_connect_loss_weight = (
                0.0 + 1.0 * manifold_loss_weight
            ) * boundary_connect_loss_weight_max

            self.trainEpoch(
                current_lr,
                gt_points,
                fit_loss_weight,
                coverage_loss_weight,
                boundary_connect_loss_weight,
            )

            if current_lr == self.min_lr:
                break

        self.autoSavePcd('final', add_idx=False)
        self.autoSaveMash('final')

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

        if self.o3d_viewer is not None:
            self.o3d_viewer.run()

        return True
