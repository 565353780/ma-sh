import numpy as np
from typing import Union
from copy import deepcopy
from trimesh import Trimesh

def combineMeshes(mesh_list: list) -> Trimesh:
    if len(mesh_list) == 0:
        print('[WARN][trimesh::combineMeshes]')
        print('\t mesh list is empty! will return an empty Trimesh!')
        return Trimesh()

    if len(mesh_list) == 1:
        return deepcopy(mesh_list[0])

    vertices_list = []
    faces_list = []

    face_start_idx = 0
    for mesh in mesh_list:
        clone_mesh = deepcopy(mesh)

        current_vertices = np.asarray(clone_mesh.vertices)
        current_faces = np.asarray(clone_mesh.faces)

        new_faces = current_faces + face_start_idx

        vertices_list.append(current_vertices)
        faces_list.append(new_faces)

        face_start_idx += current_vertices.shape[0]

    union_vertices = np.vstack(vertices_list)
    union_faces = np.vstack(faces_list)

    union_trimesh = Trimesh(vertices=union_vertices, faces=union_faces)
    return union_trimesh

def renderGeometries(geometry_list: Union[Trimesh, list]) -> bool:
    if isinstance(geometry_list, Trimesh):
        geometry_list = [geometry_list]

    union_mesh = combineMeshes(geometry_list)

    union_mesh.show()
    return True
