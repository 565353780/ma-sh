import numpy as np
import open3d as o3d
from ma_sh.Data.mesh import Mesh

gt_mesh_file_path = './gt_mesh.ply'
mash_pcd_file_path = './mash_pcd.ply'
error_max_percent = 0.1
accurate = True

gt_mesh = Mesh(gt_mesh_file_path)
mash_pcd = o3d.io.read_point_cloud(mash_pcd_file_path)

mash_pts = np.asarray(mash_pcd.points)

gt_mesh.paintJetColorsByPoints(mash_pts, error_max_percent, accurate)
