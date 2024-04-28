from typing import Union

from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, log_folder_path: Union[str, None] = None) -> None:
        self.summary_writer = None

        self.log_name_dict = {}

        if log_folder_path is not None:
            self.setLogFolder(log_folder_path)

        self.error_outputed = False
        return

    def reset(self) -> bool:
        self.summary_writer = None

        self.log_name_dict = {}
        return True

    def isValid(self) -> bool:
        return self.summary_writer is not None

    def setLogFolder(self, log_folder_path: str) -> bool:
        self.summary_writer = SummaryWriter(log_folder_path)
        return True

    def setStep(self, name: str, step: int) -> bool:
        self.log_name_dict[name] = step
        return True

    def getNameStep(self, name: str) -> int:
        if name not in self.log_name_dict.keys():
            self.setStep(name, 0)
            return 0

        return self.log_name_dict[name]

    def addStep(self, name: str) -> bool:
        return self.setStep(name, self.getNameStep(name) + 1)

    def addScalar(self, name: str, value: float, step: Union[int, None] = None) -> bool:
        if not self.isValid():
            if self.error_outputed:
                return False

            print("[ERROR][Logger::addScalar]")
            print("\t isValid failed!")
            self.error_outputed = True
            return False

        if step is not None:
            self.setStep(name, step)

        name_step = self.getNameStep(name)

        self.summary_writer.add_scalar(name, value, name_step)

        self.addStep(name)
        return True
