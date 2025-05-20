import torch
from time import time

from ma_sh.Module.trainer import Trainer
from ma_sh.Module.Convertor.base_convertor import BaseConvertor


class Convertor(BaseConvertor):
    def __init__(
        self,
        source_root_folder_path: str,
        target_root_folder_path: str,
        gt_points_num: int = 400000,
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
    ) -> None:
        super().__init__(source_root_folder_path, target_root_folder_path)

        self.gt_points_num = gt_points_num

        self.anchor_num = anchor_num
        self.mask_degree_max = mask_degree_max
        self.sh_degree_max = sh_degree_max
        self.mask_boundary_sample_num = mask_boundary_sample_num
        self.sample_point_num = sample_polar_num
        self.sample_point_scale = sample_point_scale
        self.use_inv = use_inv
        self.idx_dtype = idx_dtype
        self.dtype = dtype
        self.device = device

        self.lr = lr
        self.min_lr = min_lr
        self.warmup_step_num = warmup_step_num
        self.warmup_epoch = warmup_epoch
        self.factor = factor
        self.patience = patience

        return

    def createTrainer(self) -> Trainer:
        trainer = Trainer(
            self.anchor_num,
            self.mask_degree_max,
            self.sh_degree_max,
            self.mask_boundary_sample_num,
            self.sample_point_num,
            self.sample_point_scale,
            self.use_inv,
            self.idx_dtype,
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
            if not trainer.loadMeshFile(source_path):
                print("[ERROR][Convertor::convertData]")
                print("\t loadGTPointsFile failed!")
                return False
        except KeyboardInterrupt:
            print("[INFO][Convertor::convertData]")
            print("\t program interrupted by the user (Ctrl+C).")
            exit()
        except:
            print("[ERROR][Convertor::convertData]")
            print("\t loadGTPointsFile raise Error!")
            return False

        if not trainer.autoTrainMash(self.gt_points_num):
            print("[ERROR][Convertor::convertData]")
            print("\t autoTrainMash failed!")
            return False

        trainer.saveMashFile(target_path, True)

        print("[INFO][Convertor::convertData]")
        print("\t convert to mash spend time:", time() - start)
        return True
