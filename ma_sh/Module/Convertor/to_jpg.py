from PIL import Image

from ma_sh.Module.Convertor.base_convertor import BaseConvertor


class Convertor(BaseConvertor):
    def __init__(
        self,
        source_root_folder_path: str,
        target_root_folder_path: str,
        quality: int = 95,
    ) -> None:
        super().__init__(source_root_folder_path, target_root_folder_path)

        self.quality = quality
        return

    def convertData(self, source_path: str, target_path: str) -> bool:
        try:
            im = Image.open(source_path)
            im = im.convert('RGB')
            im.save(target_path, quality=95)
        except KeyboardInterrupt:
            print('[INFO][Convertor::convertData]')
            print('\t program interrupted by the user (Ctrl+C).')
            exit()
        except:
            print("[ERROR][Convertor::convertData]")
            print("\t convert to jpg failed!")
            print("\t source_path:", source_path)
            return False

        return True
