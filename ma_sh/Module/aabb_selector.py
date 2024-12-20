import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from ma_sh.Method.pcd import getPointCloud
from ma_sh.Model.mash import Mash


class AABBSelector:
    def __init__(self, mash: Mash):
        boundary_pts, inner_pts, inner_idxs = mash.toSamplePoints()
        self.positions = mash.positions.detach().clone().cpu().numpy()
        self.boundary_idxs = mash.mask_boundary_phi_idxs.cpu().numpy()
        self.inner_idxs = inner_idxs.detach().clone().cpu().numpy()

        self.boundary_pcd = getPointCloud(boundary_pts.detach().clone().cpu().numpy())
        self.inner_pcd = getPointCloud(inner_pts.detach().clone().cpu().numpy())
        self.selected_points = None

        gui.Application.instance.initialize()

        self.window = gui.Application.instance.create_window(
            "Mash AABB Selector", 1920, 1080)

        em = self.window.theme.font_size

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)

        self.material = rendering.MaterialRecord()
        self.material.shader = 'defaultUnlit'
        self.material.point_size = 5.0

        self.scene.scene.add_geometry("MashBoundary", self.boundary_pcd, self.material)
        self.scene.scene.add_geometry("MashInner", self.inner_pcd, self.material)

        points = np.asarray(self.boundary_pcd.points)
        aabb_min = np.min(points, axis=0)
        aabb_max = np.max(points, axis=0)

        self.sliders = []
        for i in range(3):
            slider_min = gui.Slider(gui.Slider.DOUBLE)
            slider_min.set_limits(aabb_min[i] - 0.1, aabb_max[i] + 0.1)
            slider_min.double_value = aabb_min[i] - 0.1
            slider_min.set_on_value_changed(self._on_slider_min_changed(i))

            slider_max = gui.Slider(gui.Slider.DOUBLE)
            slider_max.set_limits(aabb_min[i] - 0.1, aabb_max[i] + 0.1)
            slider_max.double_value = aabb_max[i] + 0.1
            slider_max.set_on_value_changed(self._on_slider_max_changed(i))

            self.sliders.append((slider_min, slider_max))

        self.select_button = gui.Button('Select Anchors')
        self.select_button.horizontal_padding_em = 0.5
        self.select_button.vertical_padding_em = 0
        self.select_button.set_on_clicked(self._on_select_anchors)

        aabb = self._create_aabb()
        self.scene.scene.add_geometry("AABB", aabb, self.material)
        bounds = self.boundary_pcd.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60, bounds, bounds.get_center())

        self.pannel = gui.Vert(0, gui.Margins(0.25*em, 0.25*em, 0.25*em, 0.25*em))
        self.pannel.add_fixed(0.5*em)
        for i in range(3):
            self.pannel.add_child(self.sliders[i][0])
            self.pannel.add_child(self.sliders[i][1])
            self.pannel.add_fixed(0.25*em)

        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.scene)
        self.window.add_child(self.pannel)
        self.window.add_child(self.select_button)

        self.last_selected_mask = np.zeros([mash.anchor_num], dtype=bool)
        return

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.scene.frame = r

        pannel_width = 17*layout_context.theme.font_size
        pannel_height = min(
            r.height, self.pannel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height
        )
        self.pannel.frame = gui.Rect(
            r.get_right()-pannel_width,
            r.y,
            pannel_width,
            pannel_height)

        button_pref = self.select_button.calc_preferred_size(
            layout_context, gui.Widget.Constraints())
        self.select_button.frame = gui.Rect(
            r.get_right()-pannel_width,
            r.y+pannel_height,
            pannel_width,
            button_pref.height)

    def _create_aabb(self):
        min_bound = [self.sliders[i][0].double_value for i in range(3)]
        max_bound = [self.sliders[i][1].double_value for i in range(3)]

        corners = np.array([
            [min_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]],
        ])

        lines = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        line_set.paint_uniform_color([1, 0, 0])
        return line_set

    def _update_aabb(self):
        self.scene.scene.remove_geometry("AABB")
        aabb_box = self._create_aabb()
        self.scene.scene.add_geometry("AABB", aabb_box, self.material)

        self._update_selected_anchors()

        self.scene.force_redraw()
        return True

    def _on_slider_min_changed(self, axis_index):
        def callback(value):
            self.sliders[axis_index][1].double_value = max(
                value, self.sliders[axis_index][1].double_value
            )
            self._update_aabb()
        return callback

    def _on_slider_max_changed(self, axis_index):
        def callback(value):
            self.sliders[axis_index][0].double_value = min(
                value, self.sliders[axis_index][0].double_value
            )
            self._update_aabb()
        return callback

    def _update_selected_anchors(self):
        aabb = self._create_aabb()
        mask = np.all((self.positions >= aabb.get_min_bound()) & (self.positions <= aabb.get_max_bound()), axis=1)

        if np.all(self.last_selected_mask == mask):
            return True
        self.last_selected_mask = mask

        selected_idxs = np.where(mask)[0]

        selected_boundary_pts_idxs = np.where(np.isin(self.boundary_idxs, selected_idxs))[0]
        selected_inner_pts_idxs = np.where(np.isin(self.inner_idxs, selected_idxs))[0]

        self.selected_boundary_points = self.boundary_pcd.select_by_index(selected_boundary_pts_idxs)
        self.selected_boundary_points.paint_uniform_color([0, 1, 0])
        self.selected_inner_points = self.inner_pcd.select_by_index(selected_inner_pts_idxs)
        self.selected_inner_points.paint_uniform_color([0, 1, 0])

        self.scene.scene.remove_geometry("SelectedBoundaryPoints")
        self.scene.scene.add_geometry("SelectedBoundaryPoints", self.selected_boundary_points, self.material)
        self.scene.scene.remove_geometry("SelectedInnerPoints")
        self.scene.scene.add_geometry("SelectedInnerPoints", self.selected_inner_points, self.material)
        self.scene.force_redraw()
        return True

    def _on_select_anchors(self):
        gui.Application.instance.quit()

        selected_idxs = (np.where(self.last_selected_mask)[0])

        print('[INFO][AABBSelector::on_select_anchors]')
        print('\t selected', selected_idxs.shape[0], 'anchors are:')
        print(selected_idxs)
        return True

    def run(self):
        gui.Application.instance.run()
