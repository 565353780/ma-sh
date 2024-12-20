import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from ma_sh.Model.mash import Mash


class AABBSelector:
    def __init__(self, mash: Mash):
        self.pcd = mash.toSamplePcd()
        self.selected_points = None

        # GUI Application setup
        gui.Application.instance.initialize()

        self.window = gui.Application.instance.create_window(
            "Mash AABB Selector", 1920, 1080)
        
        em = self.window.theme.font_size

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)

        self.material = rendering.MaterialRecord()
        self.material.shader = 'defaultUnlit'
        self.material.point_size = 5.0

        self.scene.scene.add_geometry("MashPcd", self.pcd, self.material)

        # Initial AABB
        points = np.asarray(self.pcd.points)
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

        self.select_button = gui.Button('Select Points')
        self.select_button.horizontal_padding_em = 0.5
        self.select_button.vertical_padding_em = 0
        self.select_button.set_on_clicked(self._select_points)

        aabb = self._create_aabb()
        self.scene.scene.add_geometry("AABB", aabb, self.material)
        bounds = self.pcd.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60, bounds, bounds.get_center())

        self.pannel = gui.Vert(0, gui.Margins(0.25*em, 0.25*em, 0.25*em, 0.25*em))
        self.pannel.add_fixed(0.5*em)
        for i in range(3):
            self.pannel.add_child(self.sliders[i][0])
            if i < 2:
                self.pannel.add_fixed(0.25*em)

        for i in range(3):
            self.pannel.add_child(self.sliders[i][1])
            if i < 2:
                self.pannel.add_fixed(0.25*em)

        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.scene)
        self.window.add_child(self.pannel)
        self.window.add_child(self.select_button)
        return

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.scene.frame = r

        pannel_width = 17*layout_context.theme.font_size
        pannel_height = min(
            r.height, self.pannel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height
        )
        self.pannel.frame = gui.Rect(r.get_right()-pannel_width,r.y,pannel_width,pannel_height)

        button_pref = self.select_button.calc_preferred_size(
            layout_context, gui.Widget.Constraints())
        self.select_button.frame = gui.Rect(r.x,r.get_bottom()-button_pref.height, button_pref.width,button_pref.height)

    def _create_aabb(self):
        min_bound = [self.sliders[i][0].double_value for i in range(3)]
        max_bound = [self.sliders[i][1].double_value for i in range(3)]

        corners = np.array([
            [min_bound[0], min_bound[1], min_bound[2]],  # 0: min x, min y, min z
            [max_bound[0], min_bound[1], min_bound[2]],  # 1: max x, min y, min z
            [max_bound[0], max_bound[1], min_bound[2]],  # 2: max x, max y, min z
            [min_bound[0], max_bound[1], min_bound[2]],  # 3: min x, max y, min z
            [min_bound[0], min_bound[1], max_bound[2]],  # 4: min x, min y, max z
            [max_bound[0], min_bound[1], max_bound[2]],  # 5: max x, min y, max z
            [max_bound[0], max_bound[1], max_bound[2]],  # 6: max x, max y, max z
            [min_bound[0], max_bound[1], max_bound[2]],  # 7: min x, max y, max z
        ])

        lines = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
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

    def _select_points(self):
        points = np.asarray(self.pcd.points)
        aabb = self._create_aabb()
        mask = np.all((points >= aabb.get_min_bound()) & (points <= aabb.get_max_bound()), axis=1)
        self.selected_points = self.pcd.select_by_index(np.where(mask)[0])
        self.selected_points.paint_uniform_color([0, 1, 0])

        # Add selected points to the scene
        self.scene.scene.remove_geometry("SelectedPoints")
        self.scene.scene.add_geometry("SelectedPoints", self.selected_points, self.material)
        self.scene.force_redraw()
        return True

    def run(self):
        gui.Application.instance.run()
