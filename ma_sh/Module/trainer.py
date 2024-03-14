import os
import torch
from typing import Union

from ma_sh.Data.mesh import Mesh
from ma_sh.Method.time import getCurrentTime
from ma_sh.Model.mash import Mash
from ma_sh.Module.logger import Logger
from ma_sh.Module.o3d_viewer import O3DViewer

class Trainer(object):
    def __init__(self,
        anchor_num: int = 4,
                 mask_degree_max: int = 5,
                 sh_degree_max: int = 3,
                 mask_boundary_sample_num: int = 10,
                 sample_polar_num: int = 10,
        idx_dtype = torch.int64,
        dtype = torch.float64,
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
        self.mash = Mash(anchor_num, mask_degree_max, sh_degree_max, mask_boundary_sample_num, sample_polar_num, idx_dtype, dtype, device)

        self.epoch = epoch

        self.step = 0
        self.loss_min = float("inf")
        self.best_params = torch.tensor([], dtype=self.mash.dtype).to(self.mash.device)

        self.params = torch.tensor([], dtype=self.mash.dtype).to(self.mash.device)

        self.lr = lr
        self.weight_decay = weight_decay
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr

        self.render = render

        self.save_folder_path = None
        self.save_file_idx = 0
        self.logger = Logger()

        self.sh_3d_degree_max = 0
        self.sh_2d_degree_max = 0
        # TODO: can start from 0 and auto upperDegrees later
        self.sh_3d_degree_max = self.mash.sh_degree_max
        self.sh_2d_degree_max = self.mash.mask_degree_max

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
