import os
import torch
from time import time
from typing import Union

from ma_sh.Method.path import createFileFolder
from ma_sh.Module.trainer import Trainer


class Convertor(object):
    def __init__(
        self,
        dataset_root_folder_path: str,
        dataset_name: str,
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
        force_start: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path

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

        self.force_start = force_start

        self.sampled_pcd_folder_path = self.dataset_root_folder_path + dataset_name + "/pcd/"
        self.mash_folder_path = self.dataset_root_folder_path + dataset_name + "/mash/"
        self.tag_folder_path = self.dataset_root_folder_path + "Tag/" + dataset_name + "_mash/"
        return

    def createTrainer(
        self,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
    ) -> Trainer:
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
            save_result_folder_path,
            save_log_folder_path,
        )

        return trainer

    def convertOneShape(self, model_id: str) -> bool:
        rel_file_path = model_id

        sampled_pcd_file_path = self.sampled_pcd_folder_path + rel_file_path + ".npy"

        if not os.path.exists(sampled_pcd_file_path):
            print("[ERROR][Convertor::convertOneShape]")
            print("\t shape file not exist!")
            print("\t sampled_pcd_file_path:", sampled_pcd_file_path)
            return False

        finish_tag_file_path = self.tag_folder_path + rel_file_path + "/finish.txt"

        if os.path.exists(finish_tag_file_path):
            return True

        start_tag_file_path = self.tag_folder_path + rel_file_path + "/start.txt"

        if os.path.exists(start_tag_file_path):
            if not self.force_start:
                return True

        createFileFolder(start_tag_file_path)

        with open(start_tag_file_path, "w") as f:
            f.write("\n")

        mash_file_path = self.mash_folder_path + rel_file_path + ".npy"

        createFileFolder(mash_file_path)

        if False:
            trainer = self.createTrainer(
                self.save_root_folder_path + "result/" + unit_rel_folder_path + "/",
                self.save_root_folder_path + "log/" + unit_rel_folder_path + "/",
            )
        else:
            trainer = self.createTrainer()

        try:
            if not trainer.loadGTPointsFile(sampled_pcd_file_path):
                print('[ERROR][Convertor::convertOneShape]')
                print('\t loadGTPointsFile failed!')
                return False
        except:
            print('[ERROR][Convertor::convertOneShape]')
            print('\t loadGTPointsFile raise Error!')
            return False

        if not trainer.autoTrainMash(self.gt_points_num):
            print('[ERROR][Convertor::convertOneShape]')
            print('\t autoTrainMash failed!')
            return False

        trainer.saveMashFile(mash_file_path, True)

        with open(finish_tag_file_path, "w") as f:
            f.write("\n")
        return True

    def convertAll(self) -> bool:
        print("[INFO][Convertor::convertAll]")
        print("\t start convert all shapes to mashes...")
        solved_shape_num = 0

        dataset_folder_path = self.sampled_pcd_folder_path

        classname_list = os.listdir(dataset_folder_path)
        classname_list.sort()

        for classname in classname_list:
            class_folder_path = dataset_folder_path + classname + "/"

            modelid_list = os.listdir(class_folder_path)
            modelid_list.sort()

            full_model_id_list = [classname + '/' + model_id[:-4] for model_id in modelid_list]

            for modelid in full_model_id_list:
                start = time()
                self.convertOneShape(modelid)
                spend = time() - start

                solved_shape_num += 1
                print("solved shape num:", solved_shape_num, 'time:', spend)
        return True
