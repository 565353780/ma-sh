import os
import torch
import trimesh
import functools
import numpy as np
import open3d as o3d
from tqdm import tqdm
from shutil import copyfile

from ma_sh.Method.pcd import getPointCloud
import mash_cpp

from ma_sh.Method.data import toNumpy
from ma_sh.Method.path import createFileFolder, removeFile, renameFile
from ma_sh.Method.render import renderGeometries

@torch.no_grad()
def compareCD():
    dataset_folder_path = '/home/chli/chLi/Dataset/'
    sample_point_num = 50000
    compare_resolution = '4000'
    dtype = torch.float32
    device = 'cuda'
    save_metric_file_path = './output/metric_manifold_sample-' + compare_resolution + '.npy'

    metric_dict = {}

    if os.path.exists(save_metric_file_path):
        metric_dict = np.load(save_metric_file_path, allow_pickle=True).item()

    gt_folder_path = dataset_folder_path + 'ManifoldMesh/'
    gt_type = '.obj'

    mash_folder_path = dataset_folder_path + 'MashV4_Recon/'
    mash_type = '.ply'

    pgr_folder_path = dataset_folder_path + 'PGR_Manifold_Recon_' + compare_resolution + '/'
    pgr_type = '.ply'

    dataset_name_list = os.listdir(mash_folder_path)
    dataset_name_list.sort()

    common_shape_num = 0

    for dataset_name in dataset_name_list:
        if dataset_name not in metric_dict.keys():
            metric_dict[dataset_name] = {}

        gt_dataset_folder_path = gt_folder_path + dataset_name + '/'

        mash_dataset_folder_path = mash_folder_path + dataset_name + '/'

        pgr_dataset_folder_path = pgr_folder_path + dataset_name + '/'
        if not os.path.exists(pgr_dataset_folder_path):
            continue

        category_list = os.listdir(mash_dataset_folder_path)
        category_list.sort()

        for category in category_list:
            if category not in metric_dict[dataset_name].keys():
                metric_dict[dataset_name][category] = {}

            gt_category_folder_path = gt_dataset_folder_path + category + '/'

            mash_category_folder_path = mash_dataset_folder_path + category + '/'

            pgr_category_folder_path = pgr_dataset_folder_path + category + '/'
            if not os.path.exists(pgr_category_folder_path):
                continue

            mash_mesh_filename_list = os.listdir(mash_category_folder_path)
            mash_mesh_filename_list.sort()

            for mesh_filename in mash_mesh_filename_list:
                mesh_id = mesh_filename.split(mash_type)[0]

                if mesh_id not in metric_dict[dataset_name][category].keys():
                    metric_dict[dataset_name][category][mesh_id] = {}

                gt_mesh_file_path = gt_category_folder_path + mesh_id + gt_type

                mash_mesh_file_path = mash_category_folder_path + mesh_id + mash_type

                pgr_mesh_file_path = pgr_category_folder_path + mesh_id + pgr_type
                if not os.path.exists(pgr_mesh_file_path):
                    continue

                common_shape_num += 1
                is_update = False

                gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_file_path)
                gt_sample_pcd = gt_mesh.sample_points_uniformly(sample_point_num)
                gt_pts = torch.from_numpy(np.asarray(gt_sample_pcd.points)).unsqueeze(0).type(dtype).to(device)

                if 'mash_cd' not in metric_dict[dataset_name][category][mesh_id].keys():
                    mash_mesh = trimesh.load_mesh(mash_mesh_file_path)
                    mash_sample_pts = mash_mesh.sample(sample_point_num)
                    mash_pts = torch.from_numpy(np.asarray(mash_sample_pts)).type(dtype).to(device)
                    mash_dist1, mash_dist2 = mash_cpp.toChamferDistanceLoss(mash_pts, gt_pts)
                    mash_cd = toNumpy(mash_dist1 + mash_dist2)
                    metric_dict[dataset_name][category][mesh_id]['mash_cd'] = mash_cd
                    is_update = True

                if 'pgr_cd' not in metric_dict[dataset_name][category][mesh_id].keys():
                    pgr_mesh = trimesh.load_mesh(pgr_mesh_file_path)
                    pgr_sample_pts = pgr_mesh.sample(sample_point_num)
                    pgr_pts = torch.from_numpy(np.asarray(pgr_sample_pts)).type(dtype).to(device)
                    pgr_dist1, pgr_dist2 = mash_cpp.toChamferDistanceLoss(pgr_pts, gt_pts)
                    pgr_cd = toNumpy(pgr_dist1 + pgr_dist2)
                    metric_dict[dataset_name][category][mesh_id]['pgr_cd'] = pgr_cd
                    is_update = True

                print(common_shape_num, '--> shape_id:', mesh_id, '\t mash_cd:', metric_dict[dataset_name][category][mesh_id]['mash_cd'], '\t pgr_cd:', metric_dict[dataset_name][category][mesh_id]['pgr_cd'])

                if is_update:
                    tmp_save_metric_file_path = save_metric_file_path.replace('.npy', '_tmp.npy')
                    createFileFolder(tmp_save_metric_file_path)

                    np.save(tmp_save_metric_file_path, metric_dict, allow_pickle=True)
                    removeFile(save_metric_file_path)
                    renameFile(tmp_save_metric_file_path, save_metric_file_path)

    return True

old_selected_id_list = [
    '31ea40d2624b40a17b431cae0dd70ed', '4cc4cb9b533e8b84b04cb542e2c50eb4', '31ea40d2624b40a17b431cae0dd70ed', '49cbfde1ae92ee555706d1c54190f27a', '22575f5719f3eb3d23bd3a8c035c98ff',
    '2ca6d10b1ba99fbd9784ddd96453bcc2', '2972fd770304663cb3d180f4523082e1', '4387affd3bc4509a36b41ce3eef1f5be', '2f7a68f4a75dabd41c8c7b57a94dbb2e', '54f13fbf4a274267a50b88953d263a42',
    '429319c0c5bddfccd26c2593d1870bdb', '3e973b624578fc89b76e29c9c43bc7aa', '35f83268d4280532dc89a28b5e6678e0', '4978fcb47c15f96dce88197ffe0c21da', '432346a3345e3e5dd79b19c7c4f0e293',
    '1b5fc54e45c8768490ad276cd2af3a4', '1e2ddaef401676915a7934ad3293bab5', '3a40eb7b9122bbfe2f066782346a992', '3c8864b07a5c0718861df5a407858f2', '42abcded68db4356352fc7e973ba7787',
    '48b257f80c7434cb56f6fc4b4ce1db04', '3799a4d787d31c0bc580fdeb5460f6d6',
]
old_ignored_id_list = [
    '2ed8d45343a442097869557127addfc0', '3bb8e6e640c32a7c36b0f2a1430e993a', '3f55885c27d84e5951ae1a6e89264401', '4bdbecfbc925219157915a20ae9ec6b6', '191360ba29f3d296ff458e602ebccbb0',
    '2882587cd2fc021c168776226d349d71', '17d7a3e8badbd881fceff3d071111703', '39de90c34ab1dd3f327289c00b6dc9ca', '33990ef5ffde80fa83bc207d8a5912e3', '45214e3010f8c3ddafa9c337f30be0ab',
    '3334213d3f378a7eacac9cf65380267d', '1bf710535121b17cf453cc5da9731a22', '2a1d80a0aa67ee7585d33ad8f24c4885', '2a2ef361fbe78f1e9f3cdee390cffc8e', '2c795b27253f9ada336961c9971d866b',
    '321f8b85785f082685e443e0ea51d93', '57263b7b778ac3a87c076cdc1de5871a', '3269630cf8dd9e87bfd5f349b1ab226', '1ee30d218c8c730ecb01bc908e8cea6', '2b9fa5950d81d925cf004563556ddb36',
    '4dd8f21d05f7d9a99e48f9f4020457c3', '18d391ede29e2edb990561fc34164364', '103a0a413d4c3353a723872ad91e4ed1', '584ab19b5df1fe697daabf84de73fb1d', '46557f689f4cf5dd2acd2bb6205825cb',
    '1b92525f3945f486fe24b6f1cb4a9319', '2c76aaa00e55c26836c07750784b6bc6', '4c1777173111f2e380a88936375f2ef4', '20a128166fc9ac939240a35c2a5f105d', '51c276e96ac4c04ebe67d9b32c3ddf8',
    '57d4b5a07e67c24af77e1de7c7a7b6e7', '432ed2387d4982a635836c728d324152', '516c4adba7205cb43e9bdff70754d92c', '373089aa7a44565b7215a1e3ffbff428', '2de1bd62aef66dc9bf65e1af50b5b7d4',
    '34d3960d35d8d5219b9f2eb77f5e247e', '493b8b8eac5d3de978f8b40f4a2ae98a', '42635d55c5905a682672e8af102e0b3c', '49748c9b987b6bc8dbd60fb6b8607ea6', '1937193cf5079b623eec26c23f5bc80b',
    '4960144353a6dd944c5483fab67d9a50', '1cf77ee00faa6de7fa6450cce25dc4cb', '2bcf0b0586570ffe6c63b8b48495261a', '2ce61518f138b7f75d009c98a5b96836', '3fdc09d3065fa3c524e7e8a625efb2a7',
    '4a2766bf63caa24a4abd5da3f5ea2512', '4b35fbf5df52938a76d876646d549aa0', '26c9e85dfa18af9fcf004563556ddb36', '30dc9d9cfbc01e19950c1f85d919ebc2', '26421cc00c639ee55a5324f7626af787',
    '52310bca00e6a3671201d487ecde379e', '2ac88058053d5c2671a782a4379556c7', '29b3a8cb9595cbe63b7fe46c7c128390', '53f01d977de993442ea98d69e91ba870', '40020a5e90c75d94a93412f1b60e6fba',
]

chair_selected_id_list = [
    '1e2ddaef401676915a7934ad3293bab5', '3a40eb7b9122bbfe2f066782346a992', '3e973b624578fc89b76e29c9c43bc7aa', '35f83268d4280532dc89a28b5e6678e0', '42abcded68db4356352fc7e973ba7787',
    '48b257f80c7434cb56f6fc4b4ce1db04', '49cbfde1ae92ee555706d1c54190f27a', '2972fd770304663cb3d180f4523082e1', '3799a4d787d31c0bc580fdeb5460f6d6',
]
airplane_selected_id_list = [
    '10c7cdfdffe2243b88a89a28f04ce622',
]
selected_id_list = chair_selected_id_list + airplane_selected_id_list
ignored_id_list = old_selected_id_list + old_ignored_id_list

@torch.no_grad()
def findBestCases():
    dataset_folder_path = '/home/chli/chLi/Dataset/'
    compare_resolution = '4000'
    save_metric_file_path = './output/metric_manifold_sample-' + compare_resolution + '.npy'
    valid_mode_list = ['selected', 'pgr-our', 'pgr-div-our', '1-div-our']
    mode_id = 0
    mode = valid_mode_list[mode_id]
    save_result_folder_path = './output/metric_manifold_result_' + mode + '/'
    save_num = 10

    os.makedirs(save_result_folder_path, exist_ok=True)

    if not os.path.exists(save_metric_file_path):
        print('[ERROR][compare::findBestCases]')
        print('\t save metric not exist!')
        print('\t save_metric_file_path:', save_metric_file_path)
        return False

    metric_dict = np.load(save_metric_file_path, allow_pickle=True).item()

    gt_mesh_folder_path = dataset_folder_path + 'ManifoldMesh/'
    gt_mesh_type = '.obj'

    gt_pcd_folder_path = dataset_folder_path + 'SampledPcd_Manifold/'
    gt_pcd_type = '.npy'

    mash_mesh_folder_path = dataset_folder_path + 'MashV4_Recon/'
    mash_mesh_type = '.ply'

    mash_pcd_folder_path = dataset_folder_path + 'MashPcd_Manifold/'
    mash_pcd_type = '.ply'

    pgr_mesh_folder_path = dataset_folder_path + 'PGR_Manifold_Recon_' + compare_resolution + '/'
    pgr_mesh_type = '.ply'

    aro_mesh_folder_path = dataset_folder_path + 'ARONet_Manifold_Recon_' + compare_resolution + '/'
    aro_mesh_type = '.obj'

    conv_mesh_folder_path = dataset_folder_path + 'ConvONet_Manifold_Recon_' + compare_resolution + '/'
    conv_mesh_type = '.obj'

    class IDWithError:
        def __init__(self, dataset_name: str, category: str, mesh_id: str, error: float) -> None:
            self.dataset_name = dataset_name
            self.category = category
            self.mesh_id = mesh_id
            self.error = error
            return

    iwe_list = []

    for dataset_name in metric_dict.keys():
        for category in metric_dict[dataset_name].keys():
            for mesh_id in metric_dict[dataset_name][category].keys():
                # mesh_id = '1b5fc54e45c8768490ad276cd2af3a4'
                mash_cd = metric_dict[dataset_name][category][mesh_id]['mash_cd']
                pgr_cd = metric_dict[dataset_name][category][mesh_id]['pgr_cd']

                if mode == 'selected':
                    error = 0
                elif mode == 'pgr-our':
                    error = pgr_cd - mash_cd
                elif mode == 'pgr-div-our':
                    error = pgr_cd / mash_cd
                elif mode == '1-div-our':
                    error = 1 / mash_cd

                iwe = IDWithError(dataset_name, category, mesh_id, error)
                iwe_list.append(iwe)

    def compare_func(x, y):
        if x.error < y.error:
            return 1
        if x.error > y.error:
            return -1
        return 0

    iwe_list.sort(key=functools.cmp_to_key(compare_func))

    saved_num = 0
    for iwe in tqdm(iwe_list):
        if mode == 'selected':
            if iwe.mesh_id not in selected_id_list:
                continue
        else:
            if selected_id_list is not None:
                if iwe.mesh_id in selected_id_list:
                    continue
            if ignored_id_list is not None:
                if iwe.mesh_id in ignored_id_list:
                    continue

        rel_file_path = iwe.dataset_name + '/' + iwe.category + '/' + iwe.mesh_id

        current_save_result_folder_path = save_result_folder_path + rel_file_path + '/'

        os.makedirs(current_save_result_folder_path, exist_ok=True)

        gt_mesh_file_path = gt_mesh_folder_path + rel_file_path + gt_mesh_type
        gt_pcd_file_path = gt_pcd_folder_path + rel_file_path + gt_pcd_type
        mash_mesh_file_path = mash_mesh_folder_path + rel_file_path + mash_mesh_type
        mash_pcd_file_path = mash_pcd_folder_path + rel_file_path + mash_pcd_type
        pgr_mesh_file_path = pgr_mesh_folder_path + rel_file_path + pgr_mesh_type
        aro_mesh_file_path = aro_mesh_folder_path + rel_file_path + aro_mesh_type
        conv_mesh_file_path = conv_mesh_folder_path + rel_file_path + conv_mesh_type

        copyfile(gt_mesh_file_path, current_save_result_folder_path + 'gt_mesh' + gt_mesh_type)

        gt_points = np.load(gt_pcd_file_path)
        gt_pcd = getPointCloud(gt_points)
        sample_gt_pcd = gt_pcd.farthest_point_down_sample(int(compare_resolution))
        o3d.io.write_point_cloud(current_save_result_folder_path + 'gt_pcd.ply', sample_gt_pcd, write_ascii=True)

        copyfile(mash_mesh_file_path, current_save_result_folder_path + 'mash_mesh' + mash_mesh_type)
        copyfile(mash_pcd_file_path, current_save_result_folder_path + 'mash_pcd' + mash_pcd_type)
        copyfile(pgr_mesh_file_path, current_save_result_folder_path + 'pgr_mesh' + pgr_mesh_type)
        if os.path.exists(aro_mesh_file_path):
            copyfile(aro_mesh_file_path, current_save_result_folder_path + 'aro_mesh' + aro_mesh_type)
        if os.path.exists(conv_mesh_file_path):
            copyfile(conv_mesh_file_path, current_save_result_folder_path + 'conv_mesh' + conv_mesh_type)

        saved_num += 1
        print('solved shape num:', saved_num)
        if mode != 'selected':
            if saved_num >= save_num:
                break

    exit()
    return True
