import torch

from ma_sh.Module.Convertor.to_trimesh import Convertor as ToTriMeshConvertor
from ma_sh.Module.Convertor.to_manifold import Convertor as ToManifoldConvertor
from ma_sh.Module.Convertor.mesh_to_mash import Convertor as MeshToMashConvertor
from ma_sh.Module.Convertor.sample_sdf import Convertor as SampleSDFConvertor
from data_convert.Module.pipeline_convertor import PipelineConvertor


def demo_convert(data_space: str, output_space: str, rel_data_path: str):
    data_type = "." + rel_data_path.split(".")[-1]
    rel_base_path = rel_data_path[: -len(data_type)]
    data_type_list = [data_type, ".obj", ".obj", ".npy"]

    convertor_list = [
        ToTriMeshConvertor(
            data_space + "mesh/",
            output_space + "trimesh/",
            include_texture=False,
            need_normalize=True,
        ),
        ToManifoldConvertor(
            output_space + "trimesh/",
            output_space + "manifold/",
            depth=8,
        ),
        MeshToMashConvertor(
            output_space + "manifold/",
            output_space + "mash/",
            anchor_num=4096,
            mask_degree_max=3,
            sh_degree_max=2,
            sample_phi_num=40,
            sample_theta_num=40,
            points_per_submesh=1024,
            dtype=torch.float32,
            device="cuda",
            lr=2e-3,
            min_lr=1e-3,
            warmup_step_num=80,
            warmup_epoch=4,
            factor=0.8,
            patience=2,
        ),
    ]

    gauss_noise = 0.0025
    noise_label = str(gauss_noise).replace(".", "_")
    sample_sdf_convertor = SampleSDFConvertor(
        data_space + "manifold/",
        output_space + "sdf_" + noise_label + "/",
        sample_sdf_point_num=250000,
        gauss_noise=gauss_noise,
    )
    pipeline_convertor = PipelineConvertor(convertor_list)

    pipeline_convertor.convertOneShape(rel_base_path, data_type_list)
    sample_sdf_convertor.convertOneShape(rel_base_path, data_type_list[2], ".npy")

    # pipeline_convertor.convertAll(data_type_list)
    return True
