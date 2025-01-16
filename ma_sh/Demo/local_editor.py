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
        # '02691156/e1225308d6c26c862b600da24e0965',
        # '02691156/577ef32f6c313a5e4ca16f43c2716caf',
        '02691156/cc9b7118034278fcb4cdad9a5bf52dd5',
        # '02691156/68f26c36ba5340ede58ca160a93fe29b',
        # '02691156/d0456644386d9149ce593c35f70d3f',
        # '02691156/4e3e46fa987d0892a185a70f269c2a41',
        # '02691156/f7110ecac70994a83820d8f180caa23a',
        '02691156/73f6ccf1468de18d381fd507da445af6',
        # '02691156/deefcd290f7f2f1a79201593c1eb4b0',
        # '02691156/7de379891610f5feaf7dd1bfd65143a9',
        '02691156/a75ab6e99a3542eb203936772104a82d',
        '02691156/6f96517661cf1b6799ed03445864bd37',
        '03001627/d3302b7fa6504cab1a461b43b8f257f',
        # '03001627/94ceeee26248a275e8e2378aa23e4253',
        '03001627/a75e83a3201cf5ac745004c6a29b0df0',
        '03001627/d3ff300de7ab36bfc8528ab560ff5e59',
        # '03001627/4c86a55673764f05597976c675750537',
        '03001627/d29445f24bbf1b1814c05b481f895c37',
        # '03001627/4c30d612b8663402492d9da2668ec34c',
    ]
    save_part_mash_folder_path = '/home/chli/chLi/Results/ma-sh/output/part_mash/'
    save_mash_file_path = '/home/chli/chLi/Results/ma-sh/output/part_mash/combined_mash.npy'
    overwrite = False

    for mash_id in mash_id_list:
        if os.path.exists(save_part_mash_folder_path + mash_id + '/'):
            continue

        local_editor = LocalEditor(device)
        local_editor.loadMashFiles([shapenet_folder_path + mash_id + '.npy'])

        combined_mash = local_editor.toCombinedMash()

        if combined_mash is None:
            print('toCombinedMash failed!')
            continue

        print('combined_mash anchors num:', combined_mash.anchor_num)

        combined_mash.saveParamsFile(save_part_mash_folder_path + mash_id + '/part_mash_anc-' + str(combined_mash.anchor_num) + '.npy', overwrite)

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
