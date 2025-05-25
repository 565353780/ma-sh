import os
import sys

sys.path.append("../wn-nc")
sys.path.append("../chamfer-distance")

from ma_sh.Method.path import createFileFolder
from ma_sh.Module.Convertor.to_trimesh import Convertor as ToTriMeshConvertor
from ma_sh.Module.Convertor.sample_pcd import Convertor as SamplePcdConvertor
from ma_sh.Module.Convertor.mash import Convertor as MASHConvertor


if __name__ == "__main__":
    glb_file_path = "/home/chli/chLi/Dataset/Objaverse_82K/glbs/000-000/0000ecca9a234cae994be239f6fec552.glb"
    obj_file_path = "/home/chli/chLi/Dataset/Objaverse_82K/test/obj/000-000/0000ecca9a234cae994be239f6fec552.obj"
    pcd_file_path = "/home/chli/chLi/Dataset/Objaverse_82K/test/pcd/000-000/0000ecca9a234cae994be239f6fec552.ply"
    mash_file_path = "/home/chli/chLi/Dataset/Objaverse_82K/test/mash/000-000/0000ecca9a234cae994be239f6fec552.npy"

    if not os.path.exists(glb_file_path):
        print("[ERROR][convert_glb_to_mash::main]")
        print("\t glb file not exist!")
        print("\t glb_file_path:", glb_file_path)
        exit()

    if not os.path.exists(obj_file_path):
        createFileFolder(obj_file_path)
        ToTriMeshConvertor("", "", include_texture=False).convertData(
            glb_file_path, obj_file_path
        )

    if not os.path.exists(pcd_file_path):
        createFileFolder(pcd_file_path)
        SamplePcdConvertor("", "", gt_points_num=400000).convertData(
            obj_file_path, pcd_file_path
        )

    if not os.path.exists(mash_file_path):
        createFileFolder(mash_file_path)
        MASHConvertor(
            "", "", gt_points_num=400000, anchor_num=4000, device="cuda"
        ).convertData(pcd_file_path, mash_file_path)
