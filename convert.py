import sys

sys.path.append("../data-convert")
sys.path.append("../chamfer-distance")
sys.path.append("../sdf-generate")

from ma_sh.Demo.Convertor.pipeline_convertor import demo as demo_convert


if __name__ == "__main__":
    """
    source data file path: data_space + rel_data_path
    processed data file path: output_space (with the same folder tree structure)
    for example:
    source data file path: '/opt/dataset/TRELLIS/mesh/category_id/0.glb'
    its data_space is '/opt/dataset/TRELLIS/mesh/'
    its rel_base_path is 'category_id/0.glb'
    the output file paths will just be like:
        output_space + 'category_id/0.obj'
        output_space + 'category_id/0.obj'
    """
    data_space = "/Users/chli/chLi/Dataset/vae-eval/"
    output_space = "/Users/chli/chLi/Dataset/vae-eval_output/"
    rel_data_path = "0.glb"

    demo_convert(data_space, output_space, rel_data_path)
