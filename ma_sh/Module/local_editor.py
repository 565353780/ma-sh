import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Union

from ma_sh.Model.mash import Mash
from ma_sh.Module.aabb_selector import AABBSelector

class LocalEditor(object):
    def __init__(self, device: str = 'cpu') -> None:
        self.device = device

        self.mash_list = []
        return

    def loadMashFile(self, mash_file_path: str) -> bool:
        if not os.path.exists(mash_file_path):
            print('[ERROR][LocalEditor::loadMashFile]')
            print('\t mash file not exist!')
            print('\t mash_file_path:', mash_file_path)
            return False

        mash = Mash.fromParamsFile(
            mash_file_path,
            40,
            100,
            1.0,
            torch.int64,
            torch.float64,
            self.device,
        )

        aabb_selector = AABBSelector(mash)
        aabb_selector.run()

        selected_anchor_idxs = np.where(aabb_selector.last_selected_mask)[0]

        self.mash_list.append([mash, selected_anchor_idxs])
        return True

    def loadMashFiles(self, mash_file_path_list: list) -> bool:
        if len(mash_file_path_list) == 0:
            print('[ERROR][LocalEditor::loadMashFiles]')
            print('\t mash file not exist!')
            return False

        for mash_file_path in mash_file_path_list:
            if not self.loadMashFile(mash_file_path):
                print('[ERROR][LocalEditor::loadMashFiles]')
                print('\t loadMashFile failed!')
                return False

        return True

    def toCombinedMash(self) -> Union[Mash, None]:
        if len(self.mash_list) == 0:
            print('[ERROR][LocalEditor::toCombinedMashParams]')
            print('\t mash file not exist!')
            return None

        anchor_num = 0
        for i in range(len(self.mash_list)):
            anchor_num += self.mash_list[i][1].shape[0]

        combined_mash = Mash.fromMash(self.mash_list[0][0], anchor_num=anchor_num)

        combined_mask_params = []
        combined_sh_params = []
        combined_rotate_vectors = []
        combined_positions = []

        for i in range(len(self.mash_list)):
            selected_mask_params = self.mash_list[i][0].mask_params[self.mash_list[i][1]]
            selected_sh_params = self.mash_list[i][0].sh_params[self.mash_list[i][1]]
            selected_rotate_vectors = self.mash_list[i][0].rotate_vectors[self.mash_list[i][1]]
            selected_positions = self.mash_list[i][0].positions[self.mash_list[i][1]]

            combined_mask_params.append(selected_mask_params)
            combined_sh_params.append(selected_sh_params)
            combined_rotate_vectors.append(selected_rotate_vectors)
            combined_positions.append(selected_positions)

        combined_mask_params = torch.vstack(combined_mask_params)
        combined_sh_params = torch.vstack(combined_sh_params)
        combined_rotate_vectors = torch.vstack(combined_rotate_vectors)
        combined_positions = torch.vstack(combined_positions)

        combined_mash.loadParams(
            combined_mask_params,
            combined_sh_params,
            combined_rotate_vectors,
            combined_positions,
        )

        return combined_mash
