from sdf_generate.Method.to_manifold import toManifold

from ma_sh.Module.Convertor.base_convertor import BaseConvertor


class Convertor(BaseConvertor):
    def __init__(
        self,
        source_root_folder_path: str,
        target_root_folder_path: str,
        depth: int = 8,
    ) -> None:
        super().__init__(source_root_folder_path, target_root_folder_path)

        self.depth = depth
        return

    def convertData(self, source_path: str, target_path: str) -> bool:
        try:
            toManifold(source_path, target_path, self.depth, False)
        except:
            print("[ERROR][Convertor::convertData]")
            print("\t toManifold failed!")
            print("\t source_path:", source_path)
            return False

        return True
