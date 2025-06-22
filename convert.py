import os
import sys

from ma_sh.Method.path import removeFolder

sys.path.append("../voxel-converter")
sys.path.append("../chamfer-distance")
sys.path.append("../data-convert")
sys.path.append("../diff-curvature")
sys.path.append("../mesh-graph-cut")
sys.path.append("../sdf-generate")

from ma_sh.Demo.pipeline_convertor import demo_convert_npz


if __name__ == "__main__":
    """
    data_space: the root folder of the shape dataset
    output_space: the root folder of the generated results
    rel_data_path: data_file_path = data_space + rel_data_path
    """
    data_space = os.environ["HOME"] + "/chLi/Dataset/ShuMei/data/"
    output_space = os.environ["HOME"] + "/chLi/Dataset/ShuMei/"
    rel_data_path = "0008dc75fb3648f2af4ca8c4d711e53e.npz"
    cuda_id = "0"

    removeFolder(output_space + "mash/")
    removeFolder("./logs/")
    removeFolder("./output/")

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    demo_convert_npz(data_space, output_space, rel_data_path)
