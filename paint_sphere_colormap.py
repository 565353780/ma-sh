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
    degree_max = 6
    pos = [0.7806189373083166, -0.22644588996123463, 0.18768562205134626, ]
    sh_params = [-66268.61468250134, 206011202.8739585, 237996022.4428588, 52216125.94591013, 208926646.540792, -77003844.16287616, 49432298.47445688, 281306183.86957175, 159273633.21512774, 59356509.0395022, -56054590.95828922, -45665030.15709066, -78880680.36192429, 37155990.98511617, 94550035.82042792, 104637288.82366937, 5202967.739197559, -27276120.413741205, -25467378.69669688, 9332490.211768147, -17632818.704620067, -35488387.71816856, 4148248.763150917, 26057834.413301595, 46267385.25138224, -4183570.7983204527, -9665143.454449799, -8121945.009544256, 4897231.604220093, 4579811.3172144005, 4754245.130836357, -7252608.299484028, -8780775.857947914, -2345269.425486402, 5042342.664374571, 13880834.755284393, -906177.8372226473, -996274.0421023541, -641273.5172479651, 613330.3494871231, 747202.0381547248, -181148.85264411804, 456250.9096940427, 755287.2164925531, -457816.1291119349, -644429.9916268274, -535415.9334852084, 189908.2994574292, 1079629.830678558, ]

    resolution = 1000
    colormap = 'jet'
    background_color = [200, 200, 200]

    gt_colored_sphere, gt_dist_min, gt_dist_max = createGTColoredSphere(
        mesh_file_path, pos, resolution, colormap, background_color=background_color)

    sh_colored_sphere = createSHColoredSphere(
        degree_max, sh_params, resolution, colormap, gt_dist_min, gt_dist_max)[0]

    # sh_colored_sphere.translate([2.5, 0, 0])

    o3d.io.write_triangle_mesh('./output/gt_colored_sphere,ply', gt_colored_sphere, write_ascii=True)
    o3d.io.write_triangle_mesh('./output/sh_colored_sphere,ply', sh_colored_sphere, write_ascii=True)

    # o3d.visualization.draw_geometries([sh_colored_sphere, gt_colored_sphere])
