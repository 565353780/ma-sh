import numpy as np
import open3d as o3d
from typing import Union


class O3DViewer(object):
    def __init__(self) -> None:
        self.vis = o3d.visualization.Visualizer()
        self.save_image = False
        return

    def createWindow(self) -> bool:
        self.vis.create_window()
        return True

    def rotateView(self, phi: float, theta: float) -> bool:
        ctr = self.vis.get_view_control()
        ctr.rotate(phi, theta)
        self.update()
        return True

    def addGeometry(
        self, geometry: o3d.geometry.Geometry, reset_bbox: bool = True
    ) -> bool:
        self.vis.add_geometry(geometry, reset_bbox)
        return True

    def addGeometries(self, geometry_list: list, reset_bbox: bool = True) -> bool:
        for geometry in geometry_list:
            self.vis.add_geometry(geometry, reset_bbox)
        return True

    def removeGeometry(
        self, geometry: o3d.geometry.Geometry, reset_bbox: bool = True
    ) -> bool:
        self.vis.remove_geometry(geometry, reset_bbox)
        return True

    def removeGeometries(self, geometry_list: list, reset_bbox: bool = True) -> bool:
        for geometry in geometry_list:
            self.vis.remove_geometry(geometry, reset_bbox)
        return True

    def replaceGeometry(
        self,
        old_geometry: o3d.geometry.Geometry,
        new_geometry: o3d.geometry.Geometry,
        reset_bbox: bool = False,
    ) -> bool:
        self.removeGeometry(old_geometry)
        self.addGeometry(new_geometry)
        return True

    def updateGeometry(self, geometry: o3d.geometry.Geometry) -> bool:
        self.vis.update_geometry(geometry)
        return True

    def clearGeometries(self) -> bool:
        self.vis.clear_geometries()
        self.vis.reset_view_point(True)
        return True

    def addLabel(self, position: Union[list, np.ndarray], label: str) -> bool:
        self.vis.add_3d_label(np.ndarray(position, dtype=float), label)
        return True

    def addLabels(self, positions: Union[list, np.ndarray], labels: list) -> bool:
        for position, label in zip(positions, labels):
            self.addLabel(position, label)
        return True

    def update(self) -> bool:
        self.vis.poll_events()
        self.vis.update_renderer()
        return True

    def run(self) -> bool:
        self.vis.run()
        return True

    def destoryWindow(self) -> bool:
        self.vis.destroy_window()
        return True
