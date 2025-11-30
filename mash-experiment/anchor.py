import numpy as np
import jax
from jax import numpy as jnp, vmap, jit, grad, jacfwd
from dataclasses import dataclass
from jaxtyping import Array, PyTree, Float, Bool
from functools import partial
import trimesh
import igl

from common import (
    w0,
    rot6d_to_R3,
    eval_sh_basis,
    eval_sh_basis_12,
    spherical_to_xyz,
    normalize,
    rotvec_n_to_z,
    rotvec_to_R3,
    discrete_S,
    continuous_S2,
    mean_curvature,
    principal_directions,
)


import polyscope as ps
from icecream import ic


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["mask_params", "sh_params_h", "pos", "rot6d"],
    meta_fields=[],
)
@dataclass
class Anchor:
    mask_params: Array
    sh_params_h: Array
    pos: Array
    rot6d: Array

    @classmethod
    def init(cls, pos, rot6d, init_dist, sh_degree=2, mask_degree=3):
        mask_params = jnp.zeros((2 * mask_degree + 1,))

        # 1 + 3 + 5
        sh_params_h = jnp.zeros(((sh_degree + 1) ** 2,))
        sh_params_h = sh_params_h.at[0].set(init_dist / w0)

        return cls(mask_params, sh_params_h, pos, rot6d)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "sample_phis",
        "sample_thetas",
        "degrees",
        "height_map",
        "inv_trans",
    ],
    meta_fields=[],
)
@dataclass
class AnchorSampleData:
    sample_phis: Array
    sample_thetas: Array
    degrees: Array
    height_map: Bool
    inv_trans: Bool

    @classmethod
    def init(
        cls,
        sh_degree=2,
        mask_degree=3,
        sample_phi_num=40,
        sample_theta_num=30,
        height_map=False,
        inv_trans=True,
    ):
        # In this convention, phi is the angle rotating along the z-axis
        # (1, n) * angle / n -> (angle / n, n)
        sample_phis = jnp.linspace(
            2 * jnp.pi / sample_phi_num, 2 * jnp.pi, sample_phi_num
        )
        sample_thetas = jnp.linspace(
            jnp.pi / sample_theta_num, jnp.pi, sample_theta_num
        )

        degrees = jnp.arange(1, mask_degree + 1)
        return cls(sample_phis, sample_thetas, degrees, height_map, inv_trans)


@jit
def unroll_spherical_coords(
    sample_data: AnchorSampleData,
) -> list[Float[Array, "n m"], Float[Array, "n m"]]:
    N = sample_data.sample_phis.shape[0]
    M = sample_data.sample_thetas.shape[0]

    sample_phis = jnp.repeat(sample_data.sample_phis[:, None], M, 1)
    sample_thetas = jnp.repeat(sample_data.sample_thetas[None, :], N, 0)
    return sample_phis, sample_thetas


@jit
def rescale_sh_params_h(sh_params_h: Float[Array, " d"]):
    w0 = jnp.maximum(sh_params_h[0], 0.02)
    # Fit through linear regression
    # scaling = 0.54140345 * w0 - 0.01070609

    # A simple heuristic works as well
    # TODO: I think the correct way is to fit a 8D Gaussian, with scaled mean and covariance.
    #   Data can be collected automated by checking the sign of deformation gradien
    scaling = 0.75 * w0

    sh_params_h = sh_params_h.at[0].set(w0)
    sh_params_h = sh_params_h.at[1:].set(scaling * jnp.tanh(sh_params_h[1:]))

    return sh_params_h


@jit
def inverse_transform(
    inv_radius: Float, anchor_samples: Float[Array, " 3"]
) -> Float[Array, " 3"]:
    inv_center = jnp.array([0.0, 0.0, -inv_radius])
    # (inv_sample_offsets - inv_center) (anchor_samples - inv_center) = (2 * inv_radius)**2
    inv_distances = (2 * inv_radius) / jnp.linalg.norm(anchor_samples - inv_center)
    inv_anchor_samples = inv_center + jnp.square(inv_distances) * (
        anchor_samples - inv_center
    )
    return inv_anchor_samples


@jit
def rigid_transform(
    pos: Float[Array, "3"], rot6d: Float[Array, "6"], sample_offset: Float[Array, " 3"]
) -> Float[Array, " 3"]:
    R = rot6d_to_R3(rot6d)
    return R @ sample_offset + pos


@jit
def ray_transform(
    sh_params_h: Float[Array, " d"],
    sample_phis: Float,
    sample_thetas: Float,
    inv_trans: Bool,
):
    sample_directions = spherical_to_xyz(sample_phis, sample_thetas)
    sh_basis = eval_sh_basis(sample_phis, sample_thetas)
    sample_distances = sh_basis @ sh_params_h
    anchor_samples = sample_directions * sample_distances
    inv_radius = sh_params_h[0] * w0
    anchor_samples = jax.lax.cond(
        inv_trans, inverse_transform, lambda x, y: y, *(inv_radius, anchor_samples)
    )
    return anchor_samples


@jit
def height_map_transform(
    sh_params_h: Float[Array, " d"],
    sample_phis: Float,
    sample_thetas: Float,
    inv_trans: Bool,
):
    sample_directions = spherical_to_xyz(sample_phis, sample_thetas)
    inv_radius = sh_params_h[0] * w0
    anchor_samples = sample_directions * inv_radius
    anchor_samples = jax.lax.cond(
        inv_trans, inverse_transform, lambda x, y: y, *(inv_radius, anchor_samples)
    )
    sh_basis_12 = eval_sh_basis_12(sample_phis, sample_thetas)
    sample_distances = sh_basis_12 @ sh_params_h[1:]
    anchor_samples += jnp.array([0.0, 0.0, sample_distances])
    return anchor_samples


@jit
def eval_sample(
    spherical_samples: Float[Array, " 2"],
    mask_params: Float[Array, " c"],
    sh_params_h: Float[Array, " d"],
    pos: Float[Array, " 3"],
    rot6d: Float[Array, " 6"],
    sample_data: AnchorSampleData,
) -> Float[Array, " 3"]:
    sample_phis = spherical_samples[0]
    sample_thetas = spherical_samples[1]

    theta_weights = jax.nn.sigmoid(
        mask_params[0]
        + mask_params[1::2] @ jnp.cos(sample_data.degrees * sample_phis)
        + mask_params[2::2] @ jnp.sin(sample_data.degrees * sample_phis)
    )
    sample_thetas = sample_thetas * theta_weights

    sh_params_h = rescale_sh_params_h(sh_params_h)
    sample = jax.lax.cond(
        sample_data.height_map,
        height_map_transform,
        ray_transform,
        *(sh_params_h, sample_phis, sample_thetas, sample_data.inv_trans),
    )
    sample = rigid_transform(pos, rot6d, sample)
    return sample


@jit
def tangents_to_normal(tangents: Float[Array, "3 2"]):
    normal = jnp.linalg.cross(tangents[:, 0], tangents[:, 1])
    return normalize(normal)


@jit
def eval_sample_first_order(
    spherical_samples: Float[Array, " 2"],
    mask_params: Float[Array, " c"],
    sh_params_h: Float[Array, " d"],
    pos: Float[Array, " 3"],
    rot6d: Float[Array, " 6"],
    sample_data: AnchorSampleData,
) -> list[Float[Array, " 3"], Float[Array, " 3"]]:
    _eval = partial(
        eval_sample,
        mask_params=mask_params,
        sh_params_h=sh_params_h,
        pos=pos,
        rot6d=rot6d,
        sample_data=sample_data,
    )

    def _eval2(x):
        y = _eval(x)
        return y, y

    tangents, sample = jacfwd(_eval2, has_aux=True)(spherical_samples)
    normal = tangents_to_normal(tangents)
    return sample, tangents, normal


@jit
def eval_sample_second_order(
    spherical_samples: Float[Array, " 2"],
    mask_params: Float[Array, " c"],
    sh_params_h: Float[Array, " d"],
    pos: Float[Array, " 3"],
    rot6d: Float[Array, " 6"],
    sample_data: AnchorSampleData,
) -> list[Float[Array, " 3"], Float[Array, " 3"], Float[Array, "3 3"]]:
    _eval = partial(
        eval_sample,
        mask_params=mask_params,
        sh_params_h=sh_params_h,
        pos=pos,
        rot6d=rot6d,
        sample_data=sample_data,
    )

    def _eval2(x):
        y = _eval(x)
        return y, y

    def _eval3(x):
        ty, y = jacfwd(_eval2, has_aux=True)(x)
        ny = tangents_to_normal(ty)
        return ny, (ny, ty, y)

    dN_dX, (normal, df_dX, sample) = jacfwd(_eval3, has_aux=True)(spherical_samples)
    S2 = continuous_S2(dN_dX, df_dX)
    return sample, df_dX, normal, S2


@jit
def eval_anchor(anchor: Anchor, sample_data: AnchorSampleData) -> Float[Array, "n m 3"]:
    sample_phis, sample_thetas = unroll_spherical_coords(sample_data)
    spherical_samples = jnp.stack([sample_phis, sample_thetas], axis=-1).reshape(-1, 2)

    _eval = partial(
        eval_sample,
        mask_params=anchor.mask_params,
        sh_params_h=anchor.sh_params_h,
        pos=anchor.pos,
        rot6d=anchor.rot6d,
        sample_data=sample_data,
    )

    anchor_samples = vmap(_eval)(spherical_samples)
    return anchor_samples


@jit
def eval_anchor_vertices(
    anchor: Anchor, sample_data: AnchorSampleData
) -> Float[Array, "n m 3"]:
    sample_phis, sample_thetas = unroll_spherical_coords(sample_data)
    spherical_samples = jnp.stack([sample_phis, sample_thetas], axis=-1).reshape(-1, 2)
    spherical_samples = jnp.vstack([spherical_samples, jnp.zeros((1, 2))])

    _eval = partial(
        eval_sample,
        mask_params=anchor.mask_params,
        sh_params_h=anchor.sh_params_h,
        pos=anchor.pos,
        rot6d=anchor.rot6d,
        sample_data=sample_data,
    )

    anchor_samples = vmap(_eval)(spherical_samples)
    return anchor_samples


@jit
def eval_anchor_first_order(
    anchor: Anchor, sample_data: AnchorSampleData
) -> list[Float[Array, "n 3"], Float[Array, "n 3"]]:
    sample_phis, sample_thetas = unroll_spherical_coords(sample_data)
    spherical_samples = jnp.stack([sample_phis, sample_thetas], axis=-1).reshape(-1, 2)

    anchor_samples, anchor_tangents, anchor_normals = vmap(
        eval_sample_first_order, in_axes=(0, None, None, None, None, None)
    )(
        spherical_samples,
        anchor.mask_params,
        anchor.sh_params_h,
        anchor.pos,
        anchor.rot6d,
        sample_data,
    )
    return anchor_samples, anchor_tangents, anchor_normals


@jit
def eval_anchor_second_order(
    anchor: Anchor, sample_data: AnchorSampleData
) -> list[Float[Array, "n 3"], Float[Array, "n 3"], Float[Array, "n 3 3"]]:
    sample_phis, sample_thetas = unroll_spherical_coords(sample_data)
    spherical_samples = jnp.stack([sample_phis, sample_thetas], axis=-1).reshape(-1, 2)

    anchor_samples, anchor_tangents, anchor_normals, anchor_S2s = vmap(
        eval_sample_second_order, in_axes=(0, None, None, None, None, None)
    )(
        spherical_samples,
        anchor.mask_params,
        anchor.sh_params_h,
        anchor.pos,
        anchor.rot6d,
        sample_data,
    )
    return anchor_samples, anchor_tangents, anchor_normals, anchor_S2s


@jit
def ring_tessellation(ring_idx1, ring_idx2):
    ring_idx1_roll = jnp.roll(ring_idx1, 1)
    ring_idx2_roll = jnp.roll(ring_idx2, 1)
    return jnp.stack(
        [
            jnp.stack(
                [
                    ring_idx1,
                    ring_idx2,
                    ring_idx1_roll,
                ],
                axis=-1,
            ),
            jnp.stack([ring_idx1_roll, ring_idx2, ring_idx2_roll], axis=-1),
        ],
        axis=1,
    ).reshape(-1, 3)


@jit
def ring_tessellation_alt(ring_idx1, ring_idx2):
    ring_idx1_roll = jnp.roll(ring_idx1, 1)
    ring_idx2_roll = jnp.roll(ring_idx2, 1)
    return jnp.stack(
        [
            jnp.stack(
                [
                    ring_idx1,
                    ring_idx2,
                    ring_idx2_roll,
                ],
                axis=-1,
            ),
            jnp.stack([ring_idx1_roll, ring_idx1, ring_idx2_roll], axis=-1),
        ],
        axis=1,
    ).reshape(-1, 3)


def eval_anchor_tessellation(sample_phi_num, sample_theta_num):
    ring_idx = sample_theta_num * jnp.arange(sample_phi_num)
    ring_idx1 = jnp.arange(sample_theta_num - 1)[:, None] + ring_idx[None, :]
    ring_idx2 = jnp.arange(1, sample_theta_num)[:, None] + ring_idx[None, :]
    ring_faces = vmap(ring_tessellation)(ring_idx1, ring_idx2).reshape(-1, 3)
    tip_faces = jnp.stack(
        [
            sample_phi_num * sample_theta_num * jnp.ones((sample_phi_num,)),
            ring_idx,
            jnp.roll(ring_idx, 1),
        ],
        axis=-1,
    )
    return jnp.vstack([ring_faces, tip_faces])


def eval_anchor_tessellation_double_covering(sample_phi_num, sample_theta_num):
    ring_idx = sample_theta_num * jnp.arange(sample_phi_num)
    ring_idx1 = jnp.arange(sample_theta_num - 1)[:, None] + ring_idx[None, :]
    ring_idx2 = jnp.arange(1, sample_theta_num)[:, None] + ring_idx[None, :]
    ring_faces = jnp.vstack(
        [
            vmap(ring_tessellation)(ring_idx1, ring_idx2).reshape(-1, 3),
            vmap(ring_tessellation_alt)(ring_idx1, ring_idx2).reshape(-1, 3),
        ]
    )
    tip_faces = jnp.stack(
        [
            sample_phi_num * sample_theta_num * jnp.ones((sample_phi_num,)),
            ring_idx,
            jnp.roll(ring_idx, 1),
        ],
        axis=-1,
    )
    return jnp.vstack([ring_faces, tip_faces])


def eval_anchor_tip_sample(anchor: Anchor, sample_data: AnchorSampleData):
    return eval_sample(
        jnp.zeros((2,)),
        anchor.mask_params,
        anchor.sh_params_h,
        anchor.pos,
        anchor.rot6d,
        sample_data,
    )


def ps_draw_anchor_samples(
    anchor: Anchor, sample_data: AnchorSampleData, tag="", radius=2e-3
):
    anchor_group = ps.create_group(f"anchor_{tag}")
    ps.register_point_cloud(f"anchor_pos_{tag}", anchor.pos[None, ...]).add_to_group(
        anchor_group
    )
    anchor_samples, _, anchor_normals, anchor_S2 = eval_anchor_second_order(
        anchor, sample_data
    )
    anchor_dirs_0, anchor_dirs_1 = vmap(principal_directions)(anchor_S2)
    pc_viz = ps.register_point_cloud(
        f"anchor_samples_{tag}", anchor_samples, radius=radius
    )
    pc_viz.add_vector_quantity("anchor_normals", anchor_normals, enabled=True)
    pc_viz.add_vector_quantity("anchor_dirs_0", anchor_dirs_0)
    pc_viz.add_vector_quantity("anchor_dirs_1", anchor_dirs_1)
    pc_viz.add_to_group(anchor_group)


@jit
def eval_mash(mash: Anchor, sample_data: AnchorSampleData) -> Float[Array, "b n 3"]:
    sample_phis, sample_thetas = unroll_spherical_coords(sample_data)
    spherical_samples = jnp.stack([sample_phis, sample_thetas], axis=-1).reshape(-1, 2)

    anchor_samples = vmap(
        vmap(eval_sample, in_axes=(0, None, None, None, None, None)),
        in_axes=(None, 0, 0, 0, 0, None),
    )(
        spherical_samples,
        mash.mask_params,
        mash.sh_params_h,
        mash.pos,
        mash.rot6d,
        sample_data,
    )
    return anchor_samples


@jit
def eval_mash_vertices(
    mash: Anchor, sample_data: AnchorSampleData
) -> Float[Array, "b n 3"]:
    sample_phis, sample_thetas = unroll_spherical_coords(sample_data)
    spherical_samples = jnp.stack([sample_phis, sample_thetas], axis=-1).reshape(-1, 2)
    spherical_samples = jnp.vstack([spherical_samples, jnp.zeros((1, 2))])

    anchor_samples = vmap(
        vmap(eval_sample, in_axes=(0, None, None, None, None, None)),
        in_axes=(None, 0, 0, 0, 0, None),
    )(
        spherical_samples,
        mash.mask_params,
        mash.sh_params_h,
        mash.pos,
        mash.rot6d,
        sample_data,
    )
    return anchor_samples


def eval_mash_tessellation(sample_phi_num, sample_theta_num, anchor_size):
    F_tess = eval_anchor_tessellation(sample_phi_num, sample_theta_num)
    F_tess = (
        F_tess[None, :, :]
        + ((sample_phi_num * sample_theta_num + 1) * jnp.arange(anchor_size))[
            :, None, None
        ]
    ).reshape(-1, 3)
    return F_tess


@jit
def eval_mash_first_order(
    mash: Anchor, sample_data: AnchorSampleData
) -> list[Float[Array, "b n 3"], Float[Array, "b n 3"]]:
    sample_phis, sample_thetas = unroll_spherical_coords(sample_data)
    spherical_samples = jnp.stack([sample_phis, sample_thetas], axis=-1).reshape(-1, 2)

    anchor_samples, anchor_tangents, anchor_normals = vmap(
        vmap(
            eval_sample_first_order,
            in_axes=(0, None, None, None, None, None),
        ),
        in_axes=(None, 0, 0, 0, 0, None),
    )(
        spherical_samples,
        mash.mask_params,
        mash.sh_params_h,
        mash.pos,
        mash.rot6d,
        sample_data,
    )
    return anchor_samples, anchor_tangents, anchor_normals


@jit
def eval_mash_second_order(
    mash: Anchor, sample_data: AnchorSampleData
) -> list[Float[Array, "b n 3"], Float[Array, "b n 3"]]:
    sample_phis, sample_thetas = unroll_spherical_coords(sample_data)
    spherical_samples = jnp.stack([sample_phis, sample_thetas], axis=-1).reshape(-1, 2)

    anchor_samples, anchor_tangents, anchor_normals, anchor_S2 = vmap(
        vmap(
            eval_sample_second_order,
            in_axes=(0, None, None, None, None, None),
        ),
        in_axes=(None, 0, 0, 0, 0, None),
    )(
        spherical_samples,
        mash.mask_params,
        mash.sh_params_h,
        mash.pos,
        mash.rot6d,
        sample_data,
    )
    return anchor_samples, anchor_tangents, anchor_normals, anchor_S2


def ps_draw_mash_samples(
    mash: Anchor,
    sample_data: AnchorSampleData,
    tag="",
    radius=2e-3,
    enable_normal=False,
):
    mash_group = ps.create_group(f"mash_{tag}")

    anchor_samples, _, anchor_normals = eval_mash_first_order(mash, sample_data)

    # R3 = vmap(rot6d_to_R3)(mash.rot6d)
    # V_viz, F_viz = vis_oct_field(R3, mash.pos, 0.02)
    # ps.register_surface_mesh(f"mash_anchor_{tag}", V_viz, F_viz)
    ps.register_point_cloud(f"anchor_center_{tag}", mash.pos, enabled=False)

    anchor_group = ps.create_group(f"anchor_group_{tag}")
    mash_group.add_child_group(anchor_group)

    anchor_size = mash.pos.shape[0]
    for i in range(anchor_size):
        pc_viz = ps.register_point_cloud(
            f"anchor_samples_{tag}_{i}", anchor_samples[i], radius=radius
        )
        pc_viz.add_vector_quantity(
            "anchor_normals", anchor_normals[i], enabled=enable_normal
        )
        pc_viz.add_to_group(anchor_group)
    anchor_group.set_hide_descendants_from_structure_lists(True)
    anchor_group.set_show_child_details(False)


def initialize_anchors_mesh(
    mesh_tri: trimesh.Trimesh,
    anchor_size: int,
    init_dist: float,
    sh_degree=2,
    mask_degree=3,
) -> Anchor:
    V_sample, Fid_sample = trimesh.sample.sample_surface_even(
        mesh_tri, anchor_size, seed=0
    )
    VN_sample = mesh_tri.face_normals[Fid_sample]
    pos = jnp.asarray(V_sample + init_dist * VN_sample)

    rotvecs = -vmap(rotvec_n_to_z)(-VN_sample)
    R3 = vmap(rotvec_to_R3)(rotvecs)
    rot6d = jnp.concat([R3[..., 0], R3[..., 1]], axis=-1)

    return vmap(Anchor.init, in_axes=(0, 0, None, None, None))(
        pos, rot6d, init_dist, sh_degree, mask_degree
    )


def initialize_anchors_pc(
    pc,
    pc_n,
    init_dist: float,
    sh_degree=2,
    mask_degree=3,
) -> Anchor:
    rotvecs = -vmap(rotvec_n_to_z)(-pc_n)
    R3 = vmap(rotvec_to_R3)(rotvecs)
    rot6d = jnp.concat([R3[..., 0], R3[..., 1]], axis=-1)

    return vmap(Anchor.init, in_axes=(0, 0, None, None, None))(
        pc, rot6d, init_dist, sh_degree, mask_degree
    )


if __name__ == "__main__":
    import optax
    from tqdm import tqdm
    import equinox as eqx
    from loss import chamfer_distance_init

    np.random.seed(0)

    pos = jnp.array([-0.75, 0.0, 0.75])
    rot6d = jnp.array([0.0, 0.0, -1.0, 1.0, 0.0, 0.0])
    init_dist = 1e-1
    sample_phi_num = 40
    sample_theta_num = 30
    anchor = Anchor.init(pos, rot6d, init_dist)
    sample_data = AnchorSampleData.init(
        sample_phi_num=sample_phi_num, sample_theta_num=sample_theta_num
    )

    V, F = igl.read_triangle_mesh("assets/terrain.obj")
    VN = igl.per_vertex_normals(V, F)

    plane_samples = jnp.array(V[F].mean(1))
    plane_S2s = vmap(discrete_S)(V[F], VN[F])
    plane_dirs_0, plane_dirs_1 = vmap(principal_directions)(plane_S2s)

    ps.init()
    pc_viz = ps.register_point_cloud("plane_samples", plane_samples)
    pc_viz.add_vector_quantity("plane_dirs_0", plane_dirs_0)
    pc_viz.add_vector_quantity("plane_dirs_1", plane_dirs_1)
    ps_draw_anchor_samples(anchor, sample_data, "init")
    # ps.show()
    # exit()

    chamfer_distance = chamfer_distance_init(
        sample_phi_num * sample_theta_num, len(plane_samples)
    )

    optim = optax.adam(2e-3)
    opt_state = optim.init(anchor)

    @jit
    def loss_func(
        anchor: Anchor,
        sample_data: AnchorSampleData,
        gt_samples: Array,
        w_chamfer: Float,
        w_curvature: Float,
    ):
        anchor_samples, _, _, anchor_S2s = eval_anchor_second_order(anchor, sample_data)
        dist1, idx1, dist2, _ = chamfer_distance(anchor_samples, gt_samples)
        loss_chamfer = w_chamfer * (dist1.mean() + dist2.mean())
        loss_curvature = (
            w_curvature * vmap(jnp.linalg.norm)(anchor_S2s - plane_S2s[idx1]).mean()
        )

        loss = loss_chamfer + loss_curvature
        loss_dict = {"loss_chamfer": loss_chamfer, "loss_curvature": loss_curvature}

        return loss, loss_dict

    @jit
    def make_step(anchor: Anchor, opt_state: PyTree, loss_cfg: PyTree):
        grads, loss_dict = grad(loss_func, has_aux=True)(
            anchor, sample_data, plane_samples, **loss_cfg
        )
        updates, opt_state = optim.update(grads, opt_state, anchor)
        anchor = optax.apply_updates(anchor, updates)
        return anchor, opt_state, loss_dict

    pbar = tqdm(range(1000), dynamic_ncols=True)
    for i in pbar:
        anchor, opt_state, loss_dict = make_step(
            anchor, opt_state, {"w_chamfer": 1, "w_curvature": 0}
        )
        pbar.set_postfix(loss_dict)

    ps_draw_anchor_samples(anchor, sample_data, "After")

    pbar = tqdm(range(1000), dynamic_ncols=True)
    for i in pbar:
        anchor, opt_state, loss_dict = make_step(
            anchor, opt_state, {"w_chamfer": 1, "w_curvature": 1e-2}
        )
        pbar.set_postfix(loss_dict)

    ps_draw_anchor_samples(anchor, sample_data, "Fine")

    eqx.tree_serialise_leaves("exp/anchor.eqx", anchor)

    ps.show()
