import sys

sys.path.append("../data-convert/")
sys.path.append("../spherical-harmonics/")

import open3d as o3d

from ma_sh.Method.sphere_colormap import (
    createSHColoredSphere,
    createGTColoredSphere
)

if __name__ == '__main__':
    mesh_file_path = '/home/chli/chLi/Dataset/Famous/normalized_mesh/bunny.ply'
    pos = [0.8059904912659458, -0.22834242231612326, 0.1101409373690635, ]

    degree_max = 2
    sh_params = [46.08983669569705, -20.83063309637327, 4.404085837804623, 51.67417244334803, -9.128532383301602, -0.8784599524317446, -5.345170303069529, 1.7964291704092699, 9.356468122517784, ]

    resolution = 1000

    sh_colored_sphere = createSHColoredSphere(degree_max, sh_params, resolution)
    gt_colored_sphere = createGTColoredSphere(mesh_file_path, pos, resolution)

    sh_colored_sphere.translate([2.5, 0, 0])

    o3d.visualization.draw_geometries([sh_colored_sphere, gt_colored_sphere])
