import torch

from ma_sh.Module.Convertor.to_trimesh import Convertor as ToTriMeshConvertor
from ma_sh.Module.Convertor.to_manifold import Convertor as ToManifoldConvertor
from ma_sh.Module.Convertor.mesh_to_mash import Convertor as MeshToMashConvertor

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
    data_space: str, output_space: str, rel_data_path: str, device: str = "cuda:0"
):
    data_type = "." + rel_data_path.split(".")[-1]
    rel_base_path = rel_data_path[: -len(data_type)]

    mesh_to_mash_convertor = MeshToMashConvertor(
        data_space,
        output_space + "mash/",
        anchor_num=8192,
        mask_degree_max=2,
        sh_degree_max=2,
        sample_phi_num=40,
        sample_theta_num=40,
        points_per_submesh=1024,
        dtype=torch.float32,
        device=device,
        lr=4.0,
        min_lr=1e-1,
        warmup_step_num=80,
        warmup_epoch=4,
        factor=0.8,
        patience=2,
    )

    mesh_to_mash_convertor.convertOneShape(rel_base_path, data_type, ".npy")

    # mesh_to_mash_convertor.convertAll(mash_data_type_list)
    return True


def demo_convert_npz(
    data_space: str, output_space: str, rel_data_path: str, device: str = "cuda:0"
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
        anchor_num=8192,
        mask_degree_max=2,
        sh_degree_max=2,
        sample_phi_num=40,
        sample_theta_num=40,
        points_per_submesh=1024,
        dtype=torch.float32,
        device=device,
        lr=4.0,
        min_lr=1e-1,
        warmup_step_num=80,
        warmup_epoch=4,
        factor=0.8,
        patience=2,
    )

    mash_pipeline_convertor = PipelineConvertor(
        [npz_to_manifold_convertor, mesh_to_mash_convertor]
    )

    mash_data_type_list = [data_type, ".ply", ".npy"]
    mash_pipeline_convertor.convertOneShape(rel_base_path, mash_data_type_list)

    # mash_pipeline_convertor.convertAll(mash_data_type_list)
    return True


def demo_convert_glb(
    data_space: str, output_space: str, rel_data_path: str, device: str = "cuda:0"
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
        anchor_num=4096,
        mask_degree_max=3,
        sh_degree_max=2,
        sample_phi_num=40,
        sample_theta_num=40,
        points_per_submesh=1024,
        dtype=torch.float32,
        device=device,
        lr=2e-3,
        min_lr=1e-3,
        warmup_step_num=80,
        warmup_epoch=4,
        factor=0.8,
        patience=2,
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
