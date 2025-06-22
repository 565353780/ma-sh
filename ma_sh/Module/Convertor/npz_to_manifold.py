from voxel_converter.converter import npz_to_ply

from data_convert.Module.base_convertor import BaseConvertor


class Convertor(BaseConvertor):
    def __init__(
        self,
        source_root_folder_path: str,
        target_root_folder_path: str,
        resolution: int = 512,
        device: str = "cpu",
    ) -> None:
        super().__init__(source_root_folder_path, target_root_folder_path)

        self.resolution = resolution
        self.device = device
        return

    def convertData(self, source_path: str, target_path: str) -> bool:
        npz_to_ply(source_path, target_path, self.resolution, self.device)
        return True
