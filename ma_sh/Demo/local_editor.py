import os

from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.local_editor import LocalEditor


def demo():
    dataset_root_folder_path = toDatasetRootPath()
    assert dataset_root_folder_path is not None

    shapenet_folder_path = dataset_root_folder_path + 'MashV4/ShapeNet/'

    device = 'cuda'
    mash_id_list = [
        '02691156/595556bad291028733de69c9cd670995',
        '02691156/166d333d38897d1513d521050081b441',
        '02691156/cc9b7118034278fcb4cdad9a5bf52dd5',
        '02691156/73f6ccf1468de18d381fd507da445af6',
        '02691156/a75ab6e99a3542eb203936772104a82d',
        '02691156/6f96517661cf1b6799ed03445864bd37',

        '02828884/bdc3a9776cd0d69b26abe89c4547d5f1',
        '02828884/fc3865756db954685896bab37ddebe7',
        '02828884/d8b87f36fde3f2f3bc5996932c1238cd',

        '03001627/d3302b7fa6504cab1a461b43b8f257f',
        '03001627/a75e83a3201cf5ac745004c6a29b0df0',
        '03001627/d3ff300de7ab36bfc8528ab560ff5e59',
        '03001627/d29445f24bbf1b1814c05b481f895c37',
        '03001627/433c6c88f1a43ab73ebe788797b18766',
        '03001627/9d9b5f5b3fd41136244d7c2690850fc2',
        '03001627/f2af2483f9fb980cb237f85c0ae7ac77',
        '03001627/3c27660aacbcf99886327adaa986dff',
        '03001627/7ae6518311bf2f66e1a0327ca4c4d5a5',
    ]

    invalid_id_list = [
        '02691156/ec9bfc810a015c3a446cb1ee5d43f75f',
        '02691156/e1225308d6c26c862b600da24e0965',
        '02691156/577ef32f6c313a5e4ca16f43c2716caf',
        '02691156/68f26c36ba5340ede58ca160a93fe29b',
        '02691156/d0456644386d9149ce593c35f70d3f',
        '02691156/4e3e46fa987d0892a185a70f269c2a41',
        '02691156/f7110ecac70994a83820d8f180caa23a',
        '02691156/deefcd290f7f2f1a79201593c1eb4b0',
        '02691156/7de379891610f5feaf7dd1bfd65143a9',

        '02828884/4fb9527f357f37f99dac46bde4c69ef2',
        '02828884/2aaf5fe631a855b4de9ff1d2442cc6d1',
        '02828884/e223e77b8db4aea17d8864caa856253b',
        '02828884/2f2fb3e4f0d9c4fe9f8ae7ed6368949c',
        '02828884/91a39f76430e6cca19420b7669e7265',
        '02828884/a9aa868b77c3769ba873941124e3356f',
        '02828884/8c387d5e8ca71237d2b12aa6a0f050b3',
        '02828884/d6075b23895c7d0880e85c92fa2351f7',

        '03001627/46c6dec63dd7c45854ca97f654da3901',
        '03001627/69a6407f21509325a04c9785f6d4e317',
        '03001627/2d1f6e50aad6d88721fbac718728a36d',
        '03001627/94ceeee26248a275e8e2378aa23e4253',
        '03001627/4c86a55673764f05597976c675750537',
        '03001627/4c30d612b8663402492d9da2668ec34c',
    ]

    valid_mash_id_list = []

    for mash_id in mash_id_list:
        if mash_id in invalid_id_list:
            continue

        valid_mash_id_list.append(mash_id)

    save_part_mash_folder_path = '/home/chli/chLi/Results/ma-sh/output/part_mash/'
    overwrite = False

    for mash_id in valid_mash_id_list:
        if os.path.exists(save_part_mash_folder_path + mash_id + '/'):
            continue

        local_editor = LocalEditor(device)
        local_editor.loadMashFiles([shapenet_folder_path + mash_id + '.npy'])

        combined_mash = local_editor.toCombinedMash()

        if combined_mash is None:
            print('toCombinedMash failed!')
            continue

        print('combined_mash anchors num:', combined_mash.anchor_num)

        save_mash_file_path = save_part_mash_folder_path + mash_id + '/part_mash_anc-' + str(combined_mash.anchor_num) + '.npy'

        combined_mash.saveParamsFile(save_mash_file_path, overwrite)

        print('combined_mash saved to', save_mash_file_path)

    '''
    combined_mash = local_editor.toCombinedMash()
    if combined_mash is None:
        print('toCombinedMash failed!')
        return False

    print('combined_mash anchors num:', combined_mash.anchor_num)

    combined_mash.saveParamsFile(save_mash_file_path, overwrite)

    print('combined_mash saved to', save_mash_file_path)
    '''

    return True
