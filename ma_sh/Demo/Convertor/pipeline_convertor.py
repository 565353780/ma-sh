import sys
sys.path.append("../sdf-generate")

from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.to_trimesh import Convertor as ToTriMeshConvertor
from ma_sh.Module.Convertor.to_manifold import Convertor as ToManifoldConvertor
from ma_sh.Module.Convertor.sample_pcd import Convertor as SamplePcdConvertor
from ma_sh.Module.Convertor.sample_sdf import Convertor as SampleSDFConvertor
from ma_sh.Module.Convertor.pipeline_convertor import PipelineConvertor


def toPCDPipeline(
    dataset_folder_name: str = 'Objaverse_82K',
    source_data_folder_name: str = 'glbs',
    source_type: str = '.glb',
) -> bool:
    dataset_root_folder_path = toDatasetRootPath()
    if dataset_root_folder_path is None:
        print('[ERROR][pipeline_convertor::toPCDPipeline]')
        print('\t toDatasetRootPath failed!')
        return False

    convertor_list = [
        ToTriMeshConvertor(
            dataset_root_folder_path + dataset_folder_name + "/" + source_data_folder_name + "/",
            dataset_root_folder_path + dataset_folder_name + "/mesh/",
            include_texture=False,
            remove_source=False,
            need_normalize=True,
        ),
        ToManifoldConvertor(
            dataset_root_folder_path + dataset_folder_name + "/mesh/",
            dataset_root_folder_path + dataset_folder_name + "/manifold/",
            depth=8,
        ),
        SamplePcdConvertor(
            dataset_root_folder_path + dataset_folder_name + '/manifold/',
            dataset_root_folder_path + dataset_folder_name + '/manifold_pcd/',
            gt_points_num=400000,
        ),
    ]

    data_type_list = [
        source_type,
        '.obj',
        '.obj',
        '.npy',
    ]

    pipeline_convertor = PipelineConvertor(convertor_list)

    pipeline_convertor.convertAll(data_type_list)
    return True

def demoToPCDObjaverse():
    return toPCDPipeline('Objaverse_82K', 'glbs', '.glb')

def demoToPCDShapeNet():
    return toPCDPipeline('ShapeNet', 'ShapeNetCore.v2', '.obj')

def toSDFPipeline(
    dataset_folder_name: str = 'Objaverse_82K',
    source_data_folder_name: str = 'glbs',
    source_type: str = '.glb',
    gauss_noise: float = 0.25,
) -> bool:
    dataset_root_folder_path = toDatasetRootPath()
    if dataset_root_folder_path is None:
        print('[ERROR][pipeline_convertor::toSDFPipeline]')
        print('\t toDatasetRootPath failed!')
        return False

    # gauss_noise = 0.25
    noise_label = str(gauss_noise).replace('.', '_')

    convertor_list = [
        ToTriMeshConvertor(
            dataset_root_folder_path + dataset_folder_name + "/" + source_data_folder_name + "/",
            dataset_root_folder_path + dataset_folder_name + "/mesh/",
            include_texture=False,
            remove_source=False,
            need_normalize=True,
        ),
        ToManifoldConvertor(
            dataset_root_folder_path + dataset_folder_name + "/mesh/",
            dataset_root_folder_path + dataset_folder_name + "/manifold/",
            depth=8,
        ),
        SampleSDFConvertor(
            dataset_root_folder_path + dataset_folder_name + '/manifold/',
            dataset_root_folder_path + dataset_folder_name + '/manifold_sdf_' + noise_label + '/',
            sample_sdf_point_num=250000,
            gauss_noise=gauss_noise,
        ),
    ]

    data_type_list = [
        source_type,
        '.obj',
        '.obj',
        '.npy',
    ]

    pipeline_convertor = PipelineConvertor(convertor_list)

    pipeline_convertor.convertAll(data_type_list)
    return True

def demoToSDFObjaverse(gauss_noise: float):
    return toSDFPipeline('Objaverse_82K', 'glbs', '.glb', gauss_noise)

def demoToSDFShapeNet(gauss_noise: float):
    return toSDFPipeline('ShapeNet', 'ShapeNetCore.v2', '.obj', gauss_noise)
