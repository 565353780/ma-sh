import sys
sys.path.append('../data-convert')

from ma_sh.Module.Convertor.mesh_to_pcd import Convertor as MeshToPcdConvertor


def demo_mesh_to_pcd(
    data_space: str,
    output_space: str,
    rel_data_path: str,
    gt_points_num: int = 400000,
    random_weight: float = -1.0,
    noise_weight: float = -1.0,
    need_normalize: bool = False,
):
    data_type = "." + rel_data_path.split(".")[-1]
    rel_base_path = rel_data_path[: -len(data_type)]

    mesh_to_pcd_convertor = MeshToPcdConvertor(
        data_space,
        output_space,
        gt_points_num=gt_points_num,
        random_weight=random_weight,
        noise_weight=noise_weight,
        need_normalize=need_normalize,
    )

    mesh_to_pcd_convertor.convertOneShape(rel_base_path, data_type, ".ply")

    # mesh_to_pcd_convertor.convertAll(mash_data_type_list)
    return True
