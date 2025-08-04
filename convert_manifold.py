from ma_sh.Module.Convertor.to_manifold import Convertor as ToManifoldConvertor

if __name__ == "__main__":
    mesh_folder_path = "/home/chli/chLi/Dataset/ShapeNet/mesh/"
    manifold_folder_path = "/home/chli/chLi/Dataset/ShapeNet/manifold/"
    source_data_type = ".obj"
    target_data_type = ".ply"
    output_freq = 1.0

    to_manifold_convertor = ToManifoldConvertor(
        mesh_folder_path,
        manifold_folder_path,
        depth=8,
    )

    to_manifold_convertor.convertAll(source_data_type, target_data_type, output_freq)
