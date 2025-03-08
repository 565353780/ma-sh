import os
import numpy as np
import open3d as o3d

def toCube():
    return o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)

def toTetrahedron():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3)/2, 0.0],
        [0.5, np.sqrt(3)/6, np.sqrt(6)/3]
    ])
    triangles = np.array([
        [0, 1, 2],
        [1, 3, 2],
        [3, 0, 2],
        [0, 3, 1],
    ])

    tetrahedron = o3d.geometry.TriangleMesh()
    tetrahedron.vertices = o3d.utility.Vector3dVector(vertices)
    tetrahedron.triangles = o3d.utility.Vector3iVector(triangles)

    return tetrahedron

if __name__ == '__main__':
    save_shape_folder_path = '/home/chli/chLi/Dataset/SimpleShapes/raw_meshes/'

    os.makedirs(save_shape_folder_path, exist_ok=True)

    cube = toCube()
    tetrahedron = toTetrahedron()

    o3d.io.write_triangle_mesh(save_shape_folder_path + 'cube.ply', cube, write_ascii=True)
    o3d.io.write_triangle_mesh(save_shape_folder_path + 'tetrahedron.ply', tetrahedron, write_ascii=True)
