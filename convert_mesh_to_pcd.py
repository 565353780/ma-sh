import os

from ma_sh.Demo.mesh_to_pcd import demo_mesh_to_pcd


if __name__ == "__main__":
    """
    data_space: the root folder of the shape dataset
    output_space: the root folder of the generated results
    rel_data_path: data_file_path = data_space + rel_data_path
    """
    home = os.environ["HOME"]
    data_space = home + "/chLi/Dataset/Bunny/bunny/reconstruction/"
    output_space = home + "/chLi/Dataset/Bunny/bunny/pcd/"
    rel_data_path = "bun_zipper.ply"

    gt_points_num = 400000
    random_weight = -1.0
    noise_weight = -1.0
    need_normalize = False

    demo_mesh_to_pcd(
        data_space,
        output_space,
        rel_data_path,
        gt_points_num=gt_points_num,
        random_weight=random_weight,
        noise_weight=noise_weight,
        need_normalize=need_normalize,
    )
