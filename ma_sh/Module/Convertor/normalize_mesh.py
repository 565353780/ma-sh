import numpy as np

from ma_sh.Data.mesh import Mesh
from ma_sh.Module.Convertor.base_convertor import BaseConvertor


class Convertor(BaseConvertor):
    def __init__(
        self,
        source_root_folder_path: str,
        target_root_folder_path: str,
    ) -> None:
        super().__init__(source_root_folder_path, target_root_folder_path)
        return

    def convertData(self, source_path: str, target_path: str) -> bool:
        mesh = Mesh(source_path)

        if not mesh.isValid():
            print("[ERROR][Convertor::convertData]")
            print("\t mesh is not valid!")
            print("\t source_path:", source_path)
            return False

        min_bound = np.min(mesh.vertices, axis=0)
        max_bound = np.max(mesh.vertices, axis=0)
        length = np.max(max_bound - min_bound)
        scale = 0.9 / length
        center = (min_bound + max_bound) / 2.0

        mesh.vertices = (mesh.vertices - center) * scale

        mesh.save(target_path, True)

        return True
