import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"

import numpy as np
import jax
from jax import numpy as jnp, jit, vmap
from jaxtyping import Array, Float
import math


# Clever trick from: https://github.com/jax-ml/jax/issues/1563
# TODO: match jax syntax
def jacdiag(f, x):
    def partial_grad_f_index(i):
        def partial_grad_f_x(x_i):
            return f(x.at[i].set(x_i))[i]

        return jax.jacrev(partial_grad_f_x)(x[i])

    return jax.vmap(partial_grad_f_index)(jnp.arange(x.shape[0]))


# In this convention, phi is the angle rotating along the z-axis
@jit
def spherical_to_xyz(phi, theta):
    return jnp.stack(
        [
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.ones_like(phi) * jnp.cos(theta),
        ],
        axis=-1,
    )


@jit
def normalize(x):
    return x / (jnp.linalg.norm(x) + 1e-8)


def aabb_compute(V, scale=0.9):
    V_aabb_max = V.max(0, keepdims=True)
    V_aabb_min = V.min(0, keepdims=True)
    V_center = 0.5 * (V_aabb_max + V_aabb_min)
    scale = (V_aabb_max - V_center).max() / scale
    return V_center, scale, (V_aabb_max - V_aabb_min)


def normalize_aabb(V, scale=0.9):
    V_center, scale, _ = aabb_compute(V, scale)
    return (V - V_center) / scale


w0 = 0.5 * jnp.sqrt(1 / jnp.pi)
w1 = jnp.sqrt(3 / (4 * jnp.pi))
w2 = 0.25 * jnp.sqrt(15 / jnp.pi)


@jit
def y_00(phi, theta):
    return w0 * jnp.ones_like(phi) * jnp.ones_like(theta)


@jit
def y_1_1(phi, theta):
    return w1 * jnp.sin(theta) * jnp.sin(phi)


@jit
def y_10(phi, theta):
    return w1 * jnp.cos(theta) * jnp.ones_like(phi)


@jit
def y_11(phi, theta):
    return w1 * jnp.sin(theta) * jnp.cos(phi)


@jit
def y_2_2(phi, theta):
    return w2 * jnp.sin(theta) ** 2 * jnp.sin(2 * phi)


@jit
def y_2_1(phi, theta):
    return w2 * jnp.sin(2 * theta) * jnp.sin(phi)


@jit
def y_20(phi, theta):
    return w2 / jnp.sqrt(3) * (3 * jnp.cos(theta) ** 2 - jnp.ones_like(phi))


@jit
def y_21(phi, theta):
    return w2 * jnp.sin(2 * theta) * jnp.cos(phi)


@jit
def y_22(phi, theta):
    return w2 * jnp.sin(theta) ** 2 * jnp.cos(2 * phi)


@jit
def eval_sh_basis(phi, theta):
    return jnp.stack(
        jax.tree.map(
            lambda f: f(phi, theta),
            [y_00, y_1_1, y_10, y_11, y_2_2, y_2_1, y_20, y_21, y_22],
        ),
        axis=-1,
    )


@jit
def eval_sh_basis_12(phi, theta):
    return jnp.stack(
        jax.tree.map(
            lambda f: f(phi, theta),
            [y_1_1, y_10, y_11, y_2_2, y_2_1, y_20, y_21, y_22],
        ),
        axis=-1,
    )


@jit
def project_SH(
    sample_phis: Float[Array, " m"],
    sample_thetas: Float[Array, " m"],
    data: Float[Array, " m"],
) -> Float[Array, " 9"]:
    sh_basis = eval_sh_basis(sample_phis, sample_thetas)
    sh_param_proj = (
        2
        * jnp.pi**2
        * (data[:, None] * sh_basis * jnp.sin(sample_thetas)[:, None]).sum(0)
        / len(data)
    )
    return sh_param_proj


@jit
def skew_symmetric3(rotvec):
    return jnp.array(
        [
            [0, -rotvec[2], rotvec[1]],
            [rotvec[2], 0, -rotvec[0]],
            [-rotvec[1], rotvec[0], 0],
        ]
    )


@jit
def rotvec_to_R3_Rodrigues(rotvec):
    rotvec_norm = jnp.linalg.norm(rotvec)
    A = skew_symmetric3(rotvec / rotvec_norm)
    return jnp.eye(3) + jnp.sin(rotvec_norm) * A + (1 - jnp.cos(rotvec_norm)) * A @ A


# First order approximation
@jit
def rotvec_to_R3_approx(rotvec):
    return jnp.eye(3) + skew_symmetric3(rotvec)


# Cheaper than matrix exponetial, but only support first order gradient
@jit
def rotvec_to_R3(rotvec):
    return jax.lax.cond(
        jnp.linalg.norm(rotvec) < 1e-8,
        rotvec_to_R3_approx,
        rotvec_to_R3_Rodrigues,
        rotvec,
    )


# Implement "On the Continuity of Rotation Representations in Neural Networks" by Zhou et al.
@jit
def rot6d_to_R3(rot6d):
    a0 = rot6d[:3]
    a1 = rot6d[3:]
    b0 = normalize(a0)
    b1 = normalize(a1 - jnp.dot(b0, a1) * b0)
    b2 = jnp.cross(b0, b1)
    return jnp.array([b0, b1, b2]).T


# Supplementary of https://dl.acm.org/doi/10.1145/2980179.2982408
# Also see: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
@jit
def rotvec_n_to_z(n):
    n = normalize(n)
    z = jnp.array([0, 0, 1])
    axis = jnp.cross(n, z)
    axis_norm = jnp.linalg.norm(axis) + 1e-8
    # sin(theta) = |n x z|, cos(theta) = n . z
    angle = jnp.arctan2(axis_norm, n[2])
    return angle * (axis / axis_norm)


# Supplementary of https://dl.acm.org/doi/10.1145/2980179.2982408
def vis_oct_field(R3s, V, size):
    V_cube = np.array(
        [
            [-1, -1, 1],
            [1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1],
            [-1, -1, -1],
            [1, -1, -1],
            [-1, 1, -1],
            [1, 1, -1],
        ]
    )

    F_cube = np.array(
        [
            [7, 6, 2],
            [2, 3, 7],
            [0, 4, 5],
            [5, 1, 0],
            [0, 2, 6],
            [6, 4, 0],
            [7, 3, 1],
            [1, 5, 7],
            [3, 2, 0],
            [0, 1, 3],
            [4, 6, 7],
            [7, 5, 4],
        ]
    )

    NV = len(V)
    F_vis = (
        np.repeat(F_cube[None, ...], NV, 0)
        + (len(V_cube) * np.arange(NV))[:, None, None]
    ).reshape(-1, 3)

    V_vis = (V[:, None, :] + np.einsum("nij,bj->nbi", R3s, size * V_cube)).reshape(
        -1, 3
    )

    return V_vis, F_vis


# https://arxiv.org/pdf/2306.10115
@jit
def discrete_S(V_tri, VN_tri):
    # Local basis
    b0 = normalize(V_tri[1] - V_tri[0])
    b1 = V_tri[2] - V_tri[0]
    b1 = normalize(b1 - jnp.dot(b0, b1) * b0)

    B = jnp.stack([b0, b1], axis=-1)
    uv = (V_tri - V_tri[0][None, :]) @ B

    dU = jnp.stack([uv[1] - uv[0], uv[2] - uv[0]], axis=-1)
    dN = jnp.stack([VN_tri[1] - VN_tri[0], VN_tri[2] - VN_tri[0]], axis=-1)
    S = dN @ jnp.linalg.inv(dU)

    sigma, U = jnp.linalg.eigh(S.T @ S)
    S = U @ jnp.diag(jnp.sqrt(sigma)) @ U.T
    S = B @ S @ B.T + 1e-4 * jnp.eye(3)
    return S


# I believe this is equivalent as above, except squared eigenvalues
@jit
def discrete_S2_alt(V_tri, VN_tri):
    dx = jnp.stack([V_tri[1] - V_tri[0], V_tri[2] - V_tri[0]], axis=-1)
    dn = jnp.stack([VN_tri[1] - VN_tri[0], VN_tri[2] - VN_tri[0]], axis=-1)

    S = dn @ jnp.linalg.inv(dx.T @ dx) @ dx.T
    return S.T @ S


@jit
def continuous_S2(dN, df):
    # dN: X \to N
    # df: X \to f(X)
    # S = dN \circ df^\dag : f(X) \to N
    S = dN @ jnp.linalg.inv(df.T @ df) @ df.T
    return S.T @ S


@jit
def mean_curvature(S):
    return 0.5 * jnp.trace(S)


@jit
def principal_directions(S):
    _, eigvecs = jnp.linalg.eigh(S)
    return eigvecs[:, 2], eigvecs[:, 1]


if __name__ == "__main__":
    import polyscope as ps
    from icecream import ic

    # For the sake of debugging, do NOT use equal dimension
    # Use (phi, theta)
    #   phi (0, 2 * pi), theta (0, pi)
    num_phi = 40
    num_theta = 30
    # Avoid sample endpoint
    sample_phis = np.linspace(2 * np.pi / num_phi, 2 * np.pi, num_phi)
    sample_thetas = np.linspace(np.pi / num_theta, np.pi, num_theta)

    def _roll(i):
        return jnp.roll(sample_phis, shift=i)

    sample_phis_unroll = vmap(_roll, out_axes=-1)(jnp.arange(num_theta))

    xyz_uniform = spherical_to_xyz(sample_phis[:, None], sample_thetas[None, :])
    xyz_spiral = spherical_to_xyz(sample_phis_unroll, sample_thetas[None, :])

    E = np.stack([np.arange(num_theta)[:-1], np.arange(num_theta)[1:]], axis=-1)

    ps.init()
    uniform_group = ps.create_group("uniform")
    for i in range(num_phi):
        ps.register_curve_network(f"uniform_{i}", xyz_uniform[i], E).add_to_group(
            uniform_group
        )
    uniform_group.set_hide_descendants_from_structure_lists(True)
    uniform_group.set_show_child_details(False)

    spiral_group = ps.create_group("spiral")
    for i in range(num_phi):
        ps.register_curve_network(f"spiral_{i}", xyz_spiral[i], E).add_to_group(
            spiral_group
        )
    spiral_group.set_hide_descendants_from_structure_lists(True)
    spiral_group.set_show_child_details(False)

    ps.show()
