import numpy as np
import jax
from jax import numpy as jnp, vmap, jit
import igl
from common import discrete_S, discrete_S2_alt, principal_directions

import polyscope as ps
from icecream import ic


if __name__ == "__main__":
    V, F = igl.read_triangle_mesh("assets/terrain.obj")
    VN = igl.per_vertex_normals(V, F)

    VF = V[F]
    VNF = VN[F]

    S = vmap(discrete_S)(VF, VNF)
    p1, p2 = vmap(principal_directions)(S)

    S_alt = vmap(discrete_S2_alt)(VF, VNF)
    p1_alt, p2_alt = vmap(principal_directions)(S_alt)

    ps.init()
    ps.register_surface_mesh("MS", V, F)
    pc_viz = ps.register_point_cloud("bary", VF.mean(1), radius=1e-4)
    pc_viz.add_vector_quantity("p1", p1, enabled=True)
    pc_viz.add_vector_quantity("p2", p2, enabled=True)
    pc_viz = ps.register_point_cloud("bary_alt", VF.mean(1), radius=1e-4)
    pc_viz.add_vector_quantity("p1_alt", p1_alt, enabled=True)
    pc_viz.add_vector_quantity("p2_alt", p2_alt, enabled=True)
    ps.show()
