import sys
sys.path.append("../sdf-generate")

from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.Convertor.to_obj import Convertor as ToObjConvertor
from ma_sh.Module.Convertor.to_manifold import Convertor as ToManifoldConvertor
from ma_sh.Module.Convertor.sample_pcd import Convertor as SamplePcdConvertor
from ma_sh.Module.Convertor.pipeline_convertor import PipelineConvertor


def demo():
    dataset_root_folder_path = toDatasetRootPath()
    if dataset_root_folder_path is None:
        print('[ERROR][pipeline_convertor::demo]')
        print('\t toDatasetRootPath failed!')
        return False

    convertor_list = [
            ToObjConvertor(
                dataset_root_folder_path + "Objaverse_82K/glbs/",
                dataset_root_folder_path + "Objaverse_82K/mesh/",
                remove_source=False,
                need_normalize=True,
            ),
            ToManifoldConvertor(
                dataset_root_folder_path + "Objaverse_82K/mesh/",
                dataset_root_folder_path + "Objaverse_82K/manifold/",
                depth=8,
            ),
            SamplePcdConvertor(
                dataset_root_folder_path + "/Objaverse_82K/manifold/",
                dataset_root_folder_path + "/Objaverse_82K/manifold_pcd/",
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
