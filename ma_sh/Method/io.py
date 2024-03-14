import os
import open3d as o3d


def loadPCDFile(pcd_file_path):
    if not os.path.exists(pcd_file_path):
        print("[ERROR][mesh::loadPCDFile]")
        print("\t pcd file not exist!")
        print("\t pcd_file_path:", pcd_file_path)
        return None

    return o3d.io.read_point_cloud(pcd_file_path)


def loadMeshFile(mesh_file_path):
    if not os.path.exists(mesh_file_path):
        print("[ERROR][mesh::loadMeshFile]")
        print("\t mesh file not exist!")
        print("\t mesh_file_path:", mesh_file_path)
        return None

    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh
