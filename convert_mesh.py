import os
import sys

sys.path.append("../chamfer-distance")
sys.path.append("../data-convert")
sys.path.append("../diff-curvature")
sys.path.append("../mesh-graph-cut")
sys.path.append("../sdf-generate")

from ma_sh.Demo.pipeline_convertor import demo_convert_mesh


if __name__ == "__main__":
    """
    data_space: the root folder of the shape dataset
    output_space: the root folder of the generated results
    rel_data_path: data_file_path = data_space + rel_data_path
    """
    data_space = os.environ["HOME"] + "/chLi/Dataset/Famous/"
    output_space = os.environ["HOME"] + "/chLi/Dataset/Famous/"
    rel_data_path = "bunny-v2.ply"
    cuda_id = "0"

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    demo_convert_mesh(data_space, output_space, rel_data_path)
