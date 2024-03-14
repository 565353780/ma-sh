import numpy as np
from copy import deepcopy

class Point(object):
    def __init__(self, x :float=0, y :float=0, z :float=0) -> None:
        self.x = x
        self.y = y
        self.z = z
        return

    @classmethod
    def from_numpy(cls, xyz: np.ndarray):
        return cls(xyz[0], xyz[1], xyz[2])

    @classmethod
    def from_list(cls, xyz: list):
        return cls(xyz[0], xyz[1], xyz[2])

    def reset(self) -> bool:
        self.x = 0
        self.y = 0
        self.z = 0
        return True

    def set(self, x: float, y: float, z: float) -> bool:
        self.x = x
        self.y = y
        self.z = z
        return True

    def numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def norm(self, ord: int=2) -> float:
        return np.linalg.norm(self.numpy(), ord=ord)

    def dist(self, point, ord: int=2) -> float:
        return np.linalg.norm(self.numpy() - point.numpy(), ord=ord)

    def copy(self):
        return deepcopy(self)

    def add(self, delta_point) -> bool:
        self.x += delta_point.x
        self.y += delta_point.y
        self.z += delta_point.z
        return True

    def multi(self, multi_value: float) -> bool:
        self.x *= multi_value
        self.y *= multi_value
        self.z *= multi_value
        return True

    def outputInfo(self, info_level: int=0) -> bool:
        start = '\t' * info_level
        print(start + '[Point]')
        print(start + '\t x:', self.x)
        print(start + '\t y:', self.y)
        print(start + '\t z:', self.z)
        return True
