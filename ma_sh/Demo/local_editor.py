from ma_sh.Config.custom_path import toDatasetRootPath
from ma_sh.Module.local_editor import LocalEditor

def demo():
    dataset_root_folder_path = toDatasetRootPath()
    assert dataset_root_folder_path is not None

    device = 'cuda'
    mash_file_path_list = [
        dataset_root_folder_path + 'MashV4/ShapeNet/03001627/1006be65e7bc937e9141f9b58470d646.npy',
        dataset_root_folder_path + 'MashV4/ShapeNet/03001627/1007e20d5e811b308351982a6e40cf41.npy',
    ]
    save_mash_file_path = './output/combined_mash.npy'
    overwrite = True

    local_editor = LocalEditor(device)
    local_editor.loadMashFiles(mash_file_path_list)
    combined_mash = local_editor.toCombinedMash()
    if combined_mash is None:
        print('toCombinedMash failed!')
        return False

    print('combined_mash anchors num:', combined_mash.anchor_num)

    combined_mash.saveParamsFile(save_mash_file_path, overwrite)

    print('combined_mash saved to', save_mash_file_path)

    return True
