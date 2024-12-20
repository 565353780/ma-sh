from ma_sh.Module.local_editor import LocalEditor

def demo():
    device = 'cuda'
    mash_file_path_list = [
        '/home/chli/Dataset/MashV4/ShapeNet/03001627/1006be65e7bc937e9141f9b58470d646.npy',
        '/home/chli/Dataset/MashV4/ShapeNet/03001627/1007e20d5e811b308351982a6e40cf41.npy',
    ]

    local_editor = LocalEditor(device)
    local_editor.loadMashFiles(mash_file_path_list)
    combined_mash = local_editor.toCombinedMash()
    if combined_mash is None:
        print('toCombinedMash failed!')
        return False

    print('combined_mash anchors num:', combined_mash.anchor_num)
    return True
