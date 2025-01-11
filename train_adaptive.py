import numpy as np
import open3d as o3d

from ma_sh.Demo.adaptive_trainer import demo as demo_train_adaptive
from ma_sh.Method.pcd import getPointCloud

if __name__ == "__main__":
    shape_id_list = [
        '02691156/1066b65c30d153e04c3a35cee92bb95b',
        "03001627/e71d05f223d527a5f91663a74ccd2338",
    ]
    gt_points_file_path_dict = {
        shape_id: '/home/chli/chLi2/Dataset/SampledPcd_Manifold/ShapeNet/' + shape_id + '.npy' for shape_id in shape_id_list
    }
    init_anchor_num_list = [50]
    max_fit_error_list = [1e-3, 1e-2, 1e-1]
    save_freq = 1

    for shape_id, gt_points_file_path in gt_points_file_path_dict.items():
        shape_id = shape_id.replace('/', '_')

        '''
        points = np.load(gt_points_file_path)
        pcd = getPointCloud(points)
        print('shape_id:', shape_id)
        o3d.visualization.draw_geometries([pcd])
        continue
        '''

        for init_anchor_num in init_anchor_num_list:
            for max_fit_error in max_fit_error_list:
                save_log_folder_path = './logs/' + shape_id + '/anchor-' + str(init_anchor_num) + '_err-' + str(max_fit_error) + '/'
                save_result_folder_path = './output/fit/' + shape_id + '/anchor-' + str(init_anchor_num) + '_err-' + str(max_fit_error) + '/'

                demo_train_adaptive(gt_points_file_path,
                                    init_anchor_num,
                                    max_fit_error,
                                    save_freq,
                                    save_log_folder_path,
                                    save_result_folder_path)
