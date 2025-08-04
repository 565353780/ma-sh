import torch
from typing import Union

from ma_sh.Module.Convertor.to_trimesh import Convertor as ToTriMeshConvertor
from ma_sh.Module.Convertor.to_manifold import Convertor as ToManifoldConvertor
from ma_sh.Module.Convertor.mesh_to_mash import Convertor as MeshToMashConvertor
from ma_sh.Module.Convertor.mash_to_pcd import Convertor as MashToPcdConvertor

try:
    from ma_sh.Module.Convertor.sample_sdf import Convertor as SampleSDFConvertor

    CONVERT_SDF = True
except Exception as e:
    """
    print("[ERROR][pipeline_convertor::import]")
    print("\t try import SampleSDFConvertor failed!")
    print("\t maybe current pc do not support pyrender with opengl!")
    print("\t will skip the sdf dataset conversion process!")
    print("\t error:")
    print(e)
    """
    CONVERT_SDF = False

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


def demo_convert_mesh(
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
    data_type = "." + rel_data_path.split(".")[-1]
    rel_base_path = rel_data_path[: -len(data_type)]

    mesh_to_mash_convertor = MeshToMashConvertor(
        data_space,
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

    mesh_to_mash_convertor.convertOneShape(rel_base_path, data_type, ".npy")

    # mesh_to_mash_convertor.convertAll(mash_data_type_list)
    return True


def demo_convert_npz(
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


def demo_convert_glb(
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
    data_type = "." + rel_data_path.split(".")[-1]
    rel_base_path = rel_data_path[: -len(data_type)]

    to_trimesh_convertor = ToTriMeshConvertor(
        data_space,
        output_space + "trimesh/",
        include_texture=False,
        need_normalize=True,
    )

    mesh_to_mash_convertor = MeshToMashConvertor(
        output_space + "trimesh/",
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

    to_manifold_convertor = ToManifoldConvertor(
        output_space + "trimesh/",
        output_space + "manifold/",
        depth=8,
    )

    mash_pipeline_convertor = PipelineConvertor(
        [to_trimesh_convertor, mesh_to_mash_convertor]
    )

    trimesh_data_type = ".obj"
    mash_data_type_list = [data_type, trimesh_data_type, ".npy"]
    mash_pipeline_convertor.convertOneShape(rel_base_path, mash_data_type_list)

    # mash_pipeline_convertor.convertAll(mash_data_type_list)

    if CONVERT_SDF:
        gauss_noise = 0.0025
        noise_label = str(gauss_noise).replace(".", "_")
        sample_sdf_convertor = SampleSDFConvertor(
            data_space + "manifold/",
            output_space + "sdf_" + noise_label + "/",
            sample_sdf_point_num=250000,
            gauss_noise=gauss_noise,
        )

        sdf_pipeline_convertor = PipelineConvertor(
            [to_manifold_convertor, sample_sdf_convertor]
        )

        sdf_data_type_list = [trimesh_data_type, ".ply", ".npy"]
        sdf_pipeline_convertor.convertOneShape(rel_base_path, sdf_data_type_list)

        # sdf_pipeline_convertor.convertAll(sdf_data_type_list)
    return True
