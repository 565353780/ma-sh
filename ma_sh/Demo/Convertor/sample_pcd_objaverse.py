import os

from ma_sh.Module.Convertor.sample_pcd_objaverse import Convertor


def demo():
    HOME = os.environ["HOME"]
    root_list = [
        '/mnt/data/jintian/chLi/Dataset/',
        HOME + '/chLi/Dataset/',
    ]
    dataset_root_folder_path = None
    for root in root_list:
        if os.path.exists(root):
            dataset_root_folder_path = root
            break

    if dataset_root_folder_path is None:
        print('[ERROR][sample_pcd_objaverse::demo]')
        print('\t dataset not found!')
        return False

    dataset_name = 'Objaverse_82K'
    gt_points_num = 400000
    force_start = False
    worker_num = 1

    convertor = Convertor(
        dataset_root_folder_path,
        dataset_name,
        gt_points_num,
        force_start,
    )

    convertor.convertAll(worker_num)
    return True
