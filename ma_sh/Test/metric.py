import os
import torch
import trimesh

import mash_cpp

mesh_folder_path = '/home/chli/chLi/Dataset/ManifoldMesh/ShapeNet/03001627/'


mesh_filename_list = os.listdir(mesh_folder_path)
mesh_filename_list.sort()

for mesh_filename in mesh_filename_list:
    mesh_file_path = mesh_folder_path + mesh_filename

    mesh = trimesh.load_mesh(mesh_file_path)

    pts1 = torch.from_numpy(mesh.sample(50000)).cuda()
    pts2 = torch.from_numpy(mesh.sample(50000)).unsqueeze(0).cuda()

    d1, d2 = mash_cpp.toChamferDistanceLoss(pts1, pts2)
    cd = d1 + d2

    print(d1, d2, cd)
    exit()
