from ma_sh.Method.paint import paintColormap


def demo():
    error_max_percent = 0.0001
    accurate = False
    overwrite = False

    mesh_name = 'bunny'

    normalized_mesh_file_path = '/home/chli/chLi/Dataset/Famous/normalized_mesh/' + mesh_name + '.ply'
    mash_result_folder_path = '/home/chli/chLi/Results/ma-sh/output/fit/fixed/' + mesh_name + '/'
    save_colored_gt_mesh_folder_path = '/home/chli/chLi/Results/ma-sh/output/fit_error_mesh/' + mesh_name + '/'

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
    # demo()
    demo_paint_dataset()
