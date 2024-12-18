import sys
sys.path.append("../sdf-generate")

from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.to_trimesh import Convertor as ToTriMeshConvertor
from ma_sh.Module.Convertor.to_manifold import Convertor as ToManifoldConvertor
from ma_sh.Module.Convertor.sample_pcd import Convertor as SamplePcdConvertor
from ma_sh.Module.Convertor.sample_sdf import Convertor as SampleSDFConvertor
from ma_sh.Module.Convertor.pipeline_convertor import PipelineConvertor


def demoGLB2PCD():
    dataset_root_folder_path = toDatasetRootPath()
    if dataset_root_folder_path is None:
        print('[ERROR][pipeline_convertor::demoGLB2PCD]')
        print('\t toDatasetRootPath failed!')
        return False

    convertor_list = [
        ToTriMeshConvertor(
            dataset_root_folder_path + "Objaverse_82K/glbs/",
            dataset_root_folder_path + "Objaverse_82K/mesh/",
            include_texture=False,
            remove_source=False,
            need_normalize=True,
        ),
        ToManifoldConvertor(
            dataset_root_folder_path + "Objaverse_82K/mesh/",
            dataset_root_folder_path + "Objaverse_82K/manifold/",
            depth=8,
        ),
        SamplePcdConvertor(
            dataset_root_folder_path + '/Objaverse_82K/manifold/',
            dataset_root_folder_path + '/Objaverse_82K/manifold_pcd/',
            gt_points_num=400000,
        ),
    ]

    data_type_list = [
        '.glb',
        '.obj',
        '.obj',
        '.npy',
    ]

    pipeline_convertor = PipelineConvertor(convertor_list)

    pipeline_convertor.convertAll(data_type_list)
    return True

def demoGLB2SDF(gauss_noise: float):
    dataset_root_folder_path = toDatasetRootPath()
    if dataset_root_folder_path is None:
        print('[ERROR][pipeline_convertor::demoGLB2SDF]')
        print('\t toDatasetRootPath failed!')
        return False

    # gauss_noise = 0.25
    noise_label = str(gauss_noise).replace('.', '_')

    convertor_list = [
        ToTriMeshConvertor(
            dataset_root_folder_path + "Objaverse_82K/glbs/",
            dataset_root_folder_path + "Objaverse_82K/mesh/",
            include_texture=False,
            remove_source=False,
            need_normalize=True,
        ),
        ToManifoldConvertor(
            dataset_root_folder_path + "Objaverse_82K/mesh/",
            dataset_root_folder_path + "Objaverse_82K/manifold/",
            depth=8,
        ),
        SampleSDFConvertor(
            dataset_root_folder_path + '/Objaverse_82K/manifold/',
            dataset_root_folder_path + '/Objaverse_82K/manifold_sdf_' + noise_label + '/',
            sample_sdf_point_num=250000,
            gauss_noise=gauss_noise,
        ),
    ]

    data_type_list = [
        '.glb',
        '.obj',
        '.obj',
        '.npy',
    ]

    pipeline_convertor = PipelineConvertor(convertor_list)

    pipeline_convertor.convertAll(data_type_list)
    return True
