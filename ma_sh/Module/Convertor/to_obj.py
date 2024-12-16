import open3d as o3d

from ma_sh.Method.path import removeFile
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
        mesh = o3d.io.read_triangle_mesh(source_path)
        o3d.io.write_triangle_mesh(target_path, mesh, write_ascii=True)

        removeFile(source_path)

        return True
