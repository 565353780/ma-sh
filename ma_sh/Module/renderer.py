import os
import torch
import open3d as o3d
import open3d.visualization.gui as gui

from ma_sh.Model.mash import Mash


class Renderer(object):
    def __init__(self) -> None:
        return

    def renderMash(self, mash: Mash) -> bool:
        app = gui.Application.instance
        app.initialize()

        mash_pcd = mash.toSamplePcd()

        anchor_positions = mash.positions.detach().clone().cpu().numpy()

        vis = o3d.visualization.O3DVisualizer("Mash with Anchor Idx", 1920, 1080)
        vis.show_settings = True
        vis.add_geometry("MashPcd", mash_pcd)
        for i in range(anchor_positions.shape[0]):
            vis.add_3d_label(anchor_positions[i], "{}".format(i))
        vis.reset_camera_to_default()

        app.add_window(vis)
        app.run()
        return True

    def renderMashFile(self, mash_file_path: str) -> bool:
        if not os.path.exists(mash_file_path):
            print('[ERROR][Renderer::renderMashFile]')
            print('\t mash file not exist!')
            print('\t mash_file_path:', mash_file_path)
            return False

        mash = Mash.fromParamsFile(
            mash_file_path,
            40,
            1,
            1.0,
            torch.int64,
            torch.float32,
            device='cuda',
        )

        if not self.renderMash(mash):
            print('[ERROR][Renderer::renderMashFile]')
            print('\t renderMash failed!')
            print('\t mash_file_path:', mash_file_path)
            return False

        return True
