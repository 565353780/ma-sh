import torch
from time import time

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
        sample_phi_num: int = 40,
        sample_theta_num: int = 40,
        points_per_submesh: int = 1024,
        dtype=torch.float32,
        device: str = "cuda",
        lr: float = 2e-3,
        min_lr: float = 1e-3,
        warmup_step_num: int = 80,
        warmup_epoch: int = 4,
        factor: float = 0.8,
        patience: int = 2,
    ) -> None:
        super().__init__(source_root_folder_path, target_root_folder_path)

        self.anchor_num = anchor_num
        self.mask_degree_max = mask_degree_max
        self.sh_degree_max = sh_degree_max
        self.sample_phi_num = sample_phi_num
        self.sample_theta_num = sample_theta_num
        self.points_per_submesh = points_per_submesh
        self.dtype = dtype
        self.device = device

        self.lr = lr
        self.min_lr = min_lr
        self.warmup_step_num = warmup_step_num
        self.warmup_epoch = warmup_epoch
        self.factor = factor
        self.patience = patience

        return

    def createTrainer(self) -> MeshTrainer:
        trainer = MeshTrainer(
            self.anchor_num,
            self.mask_degree_max,
            self.sh_degree_max,
            self.sample_phi_num,
            self.sample_theta_num,
            self.dtype,
            self.device,
            self.lr,
            self.min_lr,
            self.warmup_step_num,
            self.warmup_epoch,
            self.factor,
            self.patience,
            False,
            1,
            False,
            -1,
            None,
            None,
        )

        return trainer

    def convertData(self, source_path: str, target_path: str) -> bool:
        start = time()

        trainer = self.createTrainer()

        try:
            if not trainer.loadMeshFile(source_path, self.points_per_submesh):
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
