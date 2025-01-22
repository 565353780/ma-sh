from ma_sh.Method.paint import paintColormap


def demo():
    error_max_percent = 0.0001
    accurate = False
    overwrite = False

    shape_id = 'XiaomiSU7'
    shape_id = 'RobotArm'
    shape_id = 'Washer'
    # shape_id = 'bunny'

    if shape_id == 'XiaomiSU7':
        normalized_mesh_file_path = '/home/chli/chLi/Dataset/XiaomiSU7/normalized_mesh/Xiaomi_SU7_2024_low_mesh.ply'
        mash_result_folder_path = '/home/chli/chLi/Results/ma-sh/output/fit/fixed/XiaomiSU7/'
        save_colored_gt_mesh_folder_path = '/home/chli/chLi/Results/ma-sh/output/fit_error_mesh/XiaomiSU7/'

    elif shape_id == 'RobotArm':
        normalized_mesh_file_path = '/home/chli/chLi/Dataset/RobotArm/normalized_mesh/Rmk3.ply'
        mash_result_folder_path = '/home/chli/chLi/Results/ma-sh/output/fit/fixed/RobotArm/'
        save_colored_gt_mesh_folder_path = '/home/chli/chLi/Results/ma-sh/output/fit_error_mesh/RobotArm/'

    elif shape_id == 'Washer':
        normalized_mesh_file_path = '/home/chli/chLi/Dataset/Washer/normalized_mesh/BOSCH_WLG.ply'
        mash_result_folder_path = '/home/chli/chLi/Results/ma-sh/output/fit/fixed/Washer/'
        save_colored_gt_mesh_folder_path = '/home/chli/chLi/Results/ma-sh/output/fit_error_mesh/Washer/'

    elif shape_id == 'bunny':
        normalized_mesh_file_path = '/home/chli/chLi/Dataset/Famous/normalized_mesh/bunny.ply'
        mash_result_folder_path = '/home/chli/chLi/Results/ma-sh/output/fit/fixed/bunny/'
        save_colored_gt_mesh_folder_path = '/home/chli/chLi/Results/ma-sh/output/fit_error_mesh/bunny/'

    else:
        return False

    paintColormap(
        normalized_mesh_file_path,
        mash_result_folder_path,
        save_colored_gt_mesh_folder_path,
        error_max_percent,
        accurate,
        overwrite,
    )
    return True

def demo_paint_dataset():
    error_max_percent = 0.0001
    accurate = False
    overwrite = False

    shapenet_shape_id_list = [
        '02691156/1066b65c30d153e04c3a35cee92bb95b',
    ]
    objaverse_shape_id_list = [
        '000-091/91979ad79916460d92c7697464f2b5f4',
        '000-091/9df219962230449caa4c95a60feb0c9e',
    ]

    for shape_id in shapenet_shape_id_list:
        normalized_mesh_file_path = '/home/chli/chLi2/Dataset/NormalizedMesh/ShapeNet/' + shape_id + '.obj'
        mash_result_folder_path = '/home/chli/chLi/Results/ma-sh/output/fit/fixed/' + shape_id.replace('/', '_') + '/'
        save_colored_gt_mesh_folder_path = '/home/chli/chLi/Results/ma-sh/output/fit_error_mesh/' + shape_id.replace('/', '_') + '/'

        paintColormap(
            normalized_mesh_file_path,
            mash_result_folder_path,
            save_colored_gt_mesh_folder_path,
        )

    for shape_id in objaverse_shape_id_list:
        normalized_mesh_file_path = '/home/chli/chLi/Dataset/Objaverse_82K/manifold/' + shape_id + '.obj'
        mash_result_folder_path = '/home/chli/chLi/Results/ma-sh/output/fit/fixed/' + shape_id.replace('/', '_') + '/'
        save_colored_gt_mesh_folder_path = '/home/chli/chLi/Results/ma-sh/output/fit_error_mesh/' + shape_id.replace('/', '_') + '/'

        paintColormap(
            normalized_mesh_file_path,
            mash_result_folder_path,
            save_colored_gt_mesh_folder_path,
            error_max_percent,
            accurate,
            overwrite,
        )

    return True

if __name__ == "__main__":
    demo()
    # demo_paint_dataset()
