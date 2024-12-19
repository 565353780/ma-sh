from ma_sh.Module.renderer import Renderer

def demo():
    mash_file_path = '/home/chli/Dataset/MashV4/ShapeNet/03001627/1006be65e7bc937e9141f9b58470d646.npy'
    mash_file_path = '/home/chli/Dataset/MashV4/ShapeNet/03001627/1007e20d5e811b308351982a6e40cf41.npy'

    renderer = Renderer()

    renderer.renderMashFile(mash_file_path)
    return True
