import sys

sys.path.append("../blender-manage")

import os

from blender_manage.Module.blender_renderer import BlenderRenderer


if __name__ == "__main__":
    home = os.environ["HOME"]

    root_folder_path = home + "/chLi/Results/ma-sh/MeshTrainer/pcd/"
    save_image_folder_path = home + "/chLi/Results/ma-sh/MeshTrainer/render/"
    overwrite = False

    blender_renderer = BlenderRenderer(
        workers_per_cpu=4,
        workers_per_gpu=8,
        is_background=True,
        mute=True,
        gpu_id_list=[0],
        early_stop=False,
    )

    if not blender_renderer.isValid():
        print("[ERROR][render_fitting::__main__]")
        print("\t blender renderer is not valid!")
        exit()

    blender_renderer.renderFolders(
        root_folder_path,
        save_image_folder_path,
        overwrite,
    )

    blender_renderer.waitWorkers()
