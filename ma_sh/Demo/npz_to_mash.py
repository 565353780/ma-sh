import torch
from typing import Union

from ma_sh.Module.Convertor.mesh_to_mash import Convertor as MeshToMashConvertor

try:
    from ma_sh.Module.Convertor.npz_to_manifold import (
        Convertor as NpzToManifoldConvertor,
    )

    CONVERT_NPZ = True
except Exception as e:
    """
    print("[ERROR][pipeline_convertor::import]")
    print("\t try import NpzToManifoldConvertor failed!")
    print("\t will skip the npz dataset conversion process!")
    print("\t error:")
    print(e)
    """
    CONVERT_NPZ = False

from data_convert.Module.pipeline_convertor import PipelineConvertor


def demo_npz_to_mash(
    data_space: str,
    output_space: str,
    rel_data_path: str,
    anchor_num: int = 8192,
    mask_degree_max: int = 2,
    sh_degree_max: int = 2,
    sample_phi_num: int = 40,
    sample_theta_num: int = 40,
    points_per_submesh: int = 1024,
    dtype=torch.float32,
    device: str = "cuda",
    lr: float = 4.0,
    min_lr: float = 1e-1,
    warmup_step_num: int = 80,
    factor: float = 0.8,
    patience: int = 2,
    render: bool = False,
    render_freq: int = 1,
    render_init_only: bool = False,
    save_freq: int = -1,
    save_result_folder_path: Union[str, None] = None,
    save_log_folder_path: Union[str, None] = None,
):
    if not CONVERT_NPZ:
        print("[ERROR][pipeline_convertor::demo_convert_npz]")
        print("\t import NpzToManifoldConvertor failed!")
        return False

    data_type = "." + rel_data_path.split(".")[-1]
    rel_base_path = rel_data_path[: -len(data_type)]

    npz_to_manifold_convertor = NpzToManifoldConvertor(
        data_space,
        output_space + "manifold/",
        resolution=512,
        device=device,
    )

    mesh_to_mash_convertor = MeshToMashConvertor(
        output_space + "manifold/",
        output_space + "mash/",
        anchor_num=anchor_num,
        mask_degree_max=mask_degree_max,
        sh_degree_max=sh_degree_max,
        sample_phi_num=sample_phi_num,
        sample_theta_num=sample_theta_num,
        points_per_submesh=points_per_submesh,
        dtype=dtype,
        device=device,
        lr=lr,
        min_lr=min_lr,
        warmup_step_num=warmup_step_num,
        factor=factor,
        patience=patience,
        render=render,
        render_freq=render_freq,
        render_init_only=render_init_only,
        save_freq=save_freq,
        save_result_folder_path=save_result_folder_path,
        save_log_folder_path=save_log_folder_path,
    )

    mash_pipeline_convertor = PipelineConvertor(
        [npz_to_manifold_convertor, mesh_to_mash_convertor]
    )

    mash_data_type_list = [data_type, ".ply", ".npy"]
    mash_pipeline_convertor.convertOneShape(rel_base_path, mash_data_type_list)

    # mash_pipeline_convertor.convertAll(mash_data_type_list)
    return True
