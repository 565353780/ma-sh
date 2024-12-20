from ma_sh.Module.local_editor import LocalEditor

def demo():
    mash_file_path_list = [
        '/home/chli/Dataset/MashV4/ShapeNet/03001627/1006be65e7bc937e9141f9b58470d646.npy',
        '/home/chli/Dataset/MashV4/ShapeNet/03001627/1007e20d5e811b308351982a6e40cf41.npy',
    ]
    save_mash_file_path = './output/combined_mash.npy'
    overwrite = True
    render = True

    local_editor = LocalEditor()
    local_editor.combineMash(mash_file_path_list, save_mash_file_path, overwrite, render)
    return True
