import trimesh
import numpy as np

from ma_sh.Method.path import removeFile
from ma_sh.Module.Convertor.base_convertor import BaseConvertor


class Convertor(BaseConvertor):
    def __init__(
        self,
        source_root_folder_path: str,
        target_root_folder_path: str,
        include_texture: bool = True,
        remove_source: bool = False,
        need_normalize: bool = False,
    ) -> None:
        super().__init__(source_root_folder_path, target_root_folder_path)

        self.include_texture = include_texture
        self.remove_source = remove_source
        self.need_normalize = need_normalize
        return

    def convertData(self, source_path: str, target_path: str) -> bool:
        try:
            mesh = trimesh.load(source_path)
        except KeyboardInterrupt:
            print('[INFO][Convertor::convertData]')
            print('\t program interrupted by the user (Ctrl+C).')
            exit()
        except:
            print('[ERROR][Convertor::convertData]')
            print('\t load mesh file failed!')
            print('\t source_path:', source_path)
            return False

        if isinstance(mesh, trimesh.Scene):
            sub_mesh_list = [geometry for geometry in mesh.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
            if len(sub_mesh_list) == 0:
                print('[ERROR][Convertor::convertData]')
                print('\t the mesh file contains no mesh!')
                print('\t source_path:', source_path)
                return False

            mesh = trimesh.util.concatenate(sub_mesh_list)

        if self.need_normalize:
            min_bound = np.min(mesh.vertices, axis=0)
            max_bound = np.max(mesh.vertices, axis=0)
            length = np.max(max_bound - min_bound)
            scale = 0.9 / length
            center = (min_bound + max_bound) / 2.0

            mesh.vertices = (mesh.vertices - center) * scale

        mesh.export(target_path, include_texture=self.include_texture)

        if self.remove_source:
            removeFile(source_path)

        return True
