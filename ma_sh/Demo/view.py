import os
import functools

from ma_sh.Model.simple_mash import SimpleMash
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
    mash = Mash.fromParamsFile(mash_file_path, device="cpu")
    if isinstance(mash, SimpleMash):
        mash.sample_phi_num = 40
        mash.sample_theta_num = 40
    else:
        mash.mask_boundary_sample_num = 90
        mash.sample_polar_num = 1000
        mash.sample_point_scale = 0.8
    mash.updatePreLoadDatas()

    print("start show mash:", mash_file_path)
    mash.renderSamplePoints()
    return True


def demo_view_folder(mash_folder_path: str) -> bool:
    if mash_folder_path[-1] != "/":
        mash_folder_path += "/"

    sample_phi_num = 40
    sample_theta_num = 40

    mash_filename_list = os.listdir(mash_folder_path)
    mash_filename_list.sort()

    for mash_filename in mash_filename_list:
        mash_file_path = mash_folder_path + mash_filename

        mash = Mash.fromParamsFile(
            mash_file_path,
        )
        if isinstance(mash, SimpleMash):
            mash.sample_phi_num = sample_phi_num
            mash.sample_theta_num = sample_theta_num
        else:
            mash.mask_boundary_sample_num = 90
            mash.sample_polar_num = 1000
            mash.sample_point_scale = 0.8
        mash.updatePreLoadDatas()

        print("start show mash:", mash_file_path)
        mash.renderSamplePoints()
    return True


def demo_view_training(mash_folder_path: str, view_freq: int = 1) -> bool:
    if mash_folder_path[-1] != "/":
        mash_folder_path += "/"

    sample_phi_num = 40
    sample_theta_num = 40

    o3d_viewer = O3DViewer()
    o3d_viewer.createWindow()

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
        )
        if isinstance(mash, SimpleMash):
            mash.sample_phi_num = sample_phi_num
            mash.sample_theta_num = sample_theta_num
        else:
            mash.mask_boundary_sample_num = 90
            mash.sample_polar_num = 1000
            mash.sample_point_scale = 0.8
        mash.updatePreLoadDatas()

        pcd = mash.toSamplePcd()

        o3d_viewer.clearGeometries()
        o3d_viewer.addGeometry(pcd)

        print("now render is", i + 1, "/", len(mash_filename_list))

        o3d_viewer.update()

    o3d_viewer.run()
    return True
