import torch
from time import time
from typing import Union

from ma_sh.Module.mesh_trainer import MeshTrainer
from data_convert.Module.base_convertor import BaseConvertor


class Convertor(BaseConvertor):
    def __init__(
        self,
        source_root_folder_path: str,
        target_root_folder_path: str,
        anchor_num: int = 400,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        use_inv: bool = True,
        dtype=torch.float64,
        device: str = "cuda",
        sample_phi_num: int = 40,
        sample_theta_num: int = 40,
        dist_max: float = 1.0 / 200,
        points_per_submesh: int = 1024,
        lr: float = 2e-3,
        min_lr: float = 1e-3,
        warmup_step_num: int = 80,
        factor: float = 0.8,
        patience: int = 2,
        render: bool = False,
        render_freq: int = 1,
        render_init_only: bool = False,
        save_freq: int = -1,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
    ) -> None:
        super().__init__(source_root_folder_path, target_root_folder_path)

        self.anchor_num = anchor_num
        self.mask_degree_max = mask_degree_max
        self.sh_degree_max = sh_degree_max
        self.use_inv = use_inv
        self.dtype = dtype
        self.device = device
        self.sample_phi_num = sample_phi_num
        self.sample_theta_num = sample_theta_num
        self.dist_max = dist_max
        self.points_per_submesh = points_per_submesh

        self.lr = lr
        self.min_lr = min_lr
        self.warmup_step_num = warmup_step_num
        self.factor = factor
        self.patience = patience

        self.render = render
        self.render_freq = render_freq
        self.render_init_only = render_init_only
        self.save_freq = save_freq
        self.save_result_folder_path = save_result_folder_path
        self.save_log_folder_path = save_log_folder_path
        return

    def createTrainer(self) -> MeshTrainer:
        trainer = MeshTrainer(
            self.anchor_num,
            self.mask_degree_max,
            self.sh_degree_max,
            self.use_inv,
            self.dtype,
            self.device,
            self.sample_phi_num,
            self.sample_theta_num,
            self.lr,
            self.min_lr,
            self.warmup_step_num,
            self.factor,
            self.patience,
            self.render,
            self.render_freq,
            self.render_init_only,
            self.save_freq,
            self.save_result_folder_path,
            self.save_log_folder_path,
        )

        return trainer

    def convertData(self, source_path: str, target_path: str) -> bool:
        start = time()

        trainer = self.createTrainer()

        try:
            if not trainer.loadMeshFile(
                source_path, self.dist_max, self.points_per_submesh
            ):
                print("[ERROR][Convertor::convertData]")
                print("\t loadMeshFile failed!")
                return False
        except KeyboardInterrupt:
            print("[INFO][Convertor::convertData]")
            print("\t program interrupted by the user (Ctrl+C).")
            exit()
        except Exception as e:
            print("[ERROR][Convertor::convertData]")
            print("\t loadMeshFile raise Error!")
            print("\t error:")
            print(e)
            return False

        if not trainer.autoTrainMash():
            print("[ERROR][Convertor::convertData]")
            print("\t autoTrainMash failed!")
            return False

        trainer.saveMashFile(target_path, True)

        print("[INFO][Convertor::convertData]")
        print("\t convert to mash spend time:", time() - start)
        return True
