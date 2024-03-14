import numpy as np
import open3d as o3d

from ma_sh.Data.point import Point

class ABB(object):
    def __init__(self, min_point: Point=Point(), max_point: Point=Point()) -> None:
        self.min_point = min_point
        self.max_point = max_point

        self.diff_point = Point()

        self.update()
        return

    @classmethod
    def from_numpy(cls, minmax):
        return cls(Point.from_numpy(minmax[0]), Point.from_numpy(minmax[1]))

    def update(self) -> bool:
        self.diff_point.x = self.max_point.x - self.min_point.x
        self.diff_point.y = self.max_point.y - self.min_point.y
        self.diff_point.z = self.max_point.z - self.min_point.z
        return True

    def setMinPoint(self, min_point: Point) -> bool:
        self.min_point = min_point
        self.update()
        return True

    def setMaxPoint(self, max_point: Point) -> bool:
        self.max_point = max_point
        self.update()
        return True

    def setPosition(self, min_point: Point, max_point: Point) -> bool:
        self.min_point = min_point
        self.max_point = max_point
        self.update()
        return True

    def position(self) -> np.ndarray:
        return np.array([
            self.min_point.position(),
            self.max_point.position()
        ], dtype=float)

    def toOpen3DABB(self) -> o3d.geometry.AxisAlignedBoundingBox:
        points = o3d.utility.Vector3dVector(self.position())
        return o3d.geometry.AxisAlignedBoundingBox.create_from_points(points)
