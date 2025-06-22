import os
import functools

from ma_sh.Model.mash import Mash
from ma_sh.Module.o3d_viewer import O3DViewer


def compare(str_a: str, str_b: str) -> int:
    id_a = int(str_a.split("_")[0])
    id_b = int(str_b.split("_")[0])

    if id_a > id_b:
        return 1
    elif id_a == id_b:
        return 0
    return -1


def demo_view_mash(mash_file_path: str) -> bool:
    mash = Mash.fromParamsFile(mash_file_path, 40, 40, device="cpu")

    print("start show mash:", mash_file_path)
    mash.renderSamplePoints()
    return True


def demo_view_training() -> bool:
    sample_phi_num = 40
    sample_theta_num = 40
    view_freq = 4

    o3d_viewer = O3DViewer()
    o3d_viewer.createWindow()

    mash_root_folder_path = "./output/"

    mash_folename_list = os.listdir(mash_root_folder_path)
    mash_folename_list.sort()

    valid_mash_folder_path_list = []

    for mash_folename in mash_folename_list:
        mash_folder_path = mash_root_folder_path + mash_folename + "/"

        if not os.path.isdir(mash_folder_path) or not os.path.exists(mash_folder_path):
            continue

        valid_mash_folder_path_list.append(mash_folder_path)

    mash_folder_path = valid_mash_folder_path_list[-1]
    print("start view:", mash_folder_path)

    mash_filename_list = os.listdir(mash_folder_path)
    mash_filename_list.sort(key=functools.cmp_to_key(compare))

    for i, mash_filename in enumerate(mash_filename_list):
        if i + 1 != len(mash_filename_list):
            if (i + 1) % view_freq != 0:
                continue

        mash_file_path = mash_folder_path + mash_filename

        mash = Mash.fromParamsFile(
            mash_file_path,
            sample_phi_num,
            sample_theta_num,
        )

        pcd = mash.toSamplePcd()

        o3d_viewer.clearGeometries()
        o3d_viewer.addGeometry(pcd)

        print("now render is", i + 1, "/", len(mash_filename_list))

        o3d_viewer.update()

    o3d_viewer.run()
    return True
