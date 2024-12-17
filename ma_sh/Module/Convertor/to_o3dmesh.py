import numpy as np
import open3d as o3d

from ma_sh.Method.path import removeFile
from ma_sh.Module.Convertor.base_convertor import BaseConvertor


class Convertor(BaseConvertor):
    def __init__(
        self,
        source_root_folder_path: str,
        target_root_folder_path: str,
        remove_source: bool = False,
        need_normalize: bool = False,
    ) -> None:
        super().__init__(source_root_folder_path, target_root_folder_path)

        self.remove_source = remove_source
        self.need_normalize = need_normalize
        return

    def convertData(self, source_path: str, target_path: str) -> bool:
        try:
            mesh = o3d.io.read_triangle_mesh(source_path)
        except:
            print('[ERROR][Convertor::convertData]')
            print('\t read_triangle_mesh failed!')
            print('\t source_path:', source_path)
            return False

        if self.need_normalize:
            min_bound = np.min(mesh.vertices, axis=0)
            max_bound = np.max(mesh.vertices, axis=0)
            length = np.max(max_bound - min_bound)
            scale = 0.9 / length
            center = (min_bound + max_bound) / 2.0

            mesh.vertices = (mesh.vertices - center) * scale

        o3d.io.write_triangle_mesh(target_path, mesh, write_ascii=True)

        if self.remove_source:
            removeFile(source_path)

        return True
