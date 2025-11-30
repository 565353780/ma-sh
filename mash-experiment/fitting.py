import numpy as np
import jax
from jax import numpy as jnp, vmap, jit, grad
from jaxtyping import Array, PyTree, Float
import equinox as eqx
import optax
from tqdm import tqdm

import trimesh
from anchor import (
    initialize_anchors_mesh,
    eval_mash,
    eval_mash_first_order,
    eval_mash_second_order,
    ps_draw_mash_samples,
    Anchor,
    AnchorSampleData,
)
from common import discrete_S, rotvec_n_to_z, rotvec_to_R3, normalize, normalize_aabb
from loss import chamfer_distance_init

import os
from glob import glob
import shutil
import argparse

import polyscope as ps
import polyscope.imgui as psim
from icecream import ic

# Enable caching
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)


class MashVizHelper:
    def register_mash(self, mash: Anchor, sample_data: AnchorSampleData, radius=2e-3):
        anchor_samples = eval_mash(mash, sample_data)

        mash_group = ps.create_group("mash")
        anchor_group = ps.create_group("anchor_group")
        mash_group.add_child_group(anchor_group)

        self.anchor_size = mash.pos.shape[0]
        self.anchor_viz_list = []
        for i in range(self.anchor_size):
            anchor_viz = ps.register_point_cloud(
                f"anchor_samples_{i}", anchor_samples[i], radius=radius
            )
            anchor_viz.add_to_group(anchor_group)
            self.anchor_viz_list.append(anchor_viz)
        anchor_group.set_hide_descendants_from_structure_lists(True)
        anchor_group.set_show_child_details(False)

        self.pos_viz = ps.register_point_cloud("anchor_center", mash.pos, enabled=False)

    def update_mash(
        self,
        mash: Anchor,
        sample_data: AnchorSampleData,
    ):
        anchor_samples = eval_mash(mash, sample_data)
        for i in range(self.anchor_size):
            anchor_viz = self.anchor_viz_list[i]
            anchor_viz.update_point_positions(anchor_samples[i])
        self.pos_viz.update_point_positions(mash.pos)


class CheckPointer:
    def __init__(self, exp_dir="exp"):
        self.exp_dir = exp_dir

    def backup(self, save_dir):
        backup_dir = os.path.join(save_dir, "backup")
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        file_list = glob("**/*.py", recursive=True)
        for file_path in file_list:
            file_save_path = os.path.join(backup_dir, file_path)

            # Avoid recursive save
            if file_path.split("/")[0] == self.exp_dir:
                continue

            shutil.copy(file_path, file_save_path)

    def serialize(self, model: PyTree, tag):
        save_dir = os.path.join(self.exp_dir, tag)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.backup(save_dir)
        eqx.tree_serialise_leaves(os.path.join(save_dir, "model.eqx"), model)

    def deserialize(self, model: PyTree, tag) -> PyTree:
        save_dir = os.path.join(self.exp_dir, tag)
        return eqx.tree_deserialise_leaves(os.path.join(save_dir, "model.eqx"), model)


@jit
def proj_vec(v, n):
    return v - jnp.dot(v, n) * n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Evaluate only")
    args = parser.parse_args()

    np.random.seed(0)

    anchor_size = 10
    sample_phi_num = 40
    sample_theta_num = 30
    init_dist = 5e-2

    mesh_tri: trimesh.Trimesh = trimesh.load("assets/bunny.obj")
    V = jnp.array(mesh_tri.vertices)
    F = jnp.array(mesh_tri.faces)
    VN = jnp.array(mesh_tri.vertex_normals)
    FN = jnp.array(mesh_tri.face_normals)
    S2s = vmap(discrete_S)(V[F], VN[F])

    mash = initialize_anchors_mesh(mesh_tri, anchor_size, init_dist)
    sample_data = AnchorSampleData.init(
        sample_phi_num=sample_phi_num, sample_theta_num=sample_theta_num
    )

    ps.init()
    ps.register_surface_mesh(
        "Bunny", mesh_tri.vertices, mesh_tri.faces, transparency=0.25
    )

    # num_gt_samples = 10000
    # chamfer_distance = chamfer_distance_init(sample_phi_num, num_gt_samples)

    # anchor_samples, anchor_tangents, _ = eval_mash_first_order(mash, sample_data)
    # anchor_samples = anchor_samples[8][boundary_mask]
    # anchor_tangents = anchor_tangents[8][boundary_mask]
    # anchor_tangent = anchor_tangents[..., 1]

    # gt_samples, fid = trimesh.sample.sample_surface_even(mesh_tri, num_gt_samples)
    # gt_samples = jnp.asarray(gt_samples)

    # dist1, idx1, dist2, idx2 = chamfer_distance(anchor_samples, gt_samples)
    # target_samples = gt_samples[idx1]
    # target_normals = FN[fid][idx1]
    # target_S2s = S2s[fid][idx1]

    # target_tangents = vmap(proj_vec)(anchor_tangent, target_normals)
    # target_tangents = vmap(normalize)(target_tangents)
    # target_tangents = jax.lax.stop_gradient(target_tangents)

    # kappa = (
    #     jnp.einsum("nij,nj->ni", target_S2s, target_tangents) * target_tangents
    # ).sum(-1)

    # ic(target_tangents)

    # t = 0.05
    # target_samples_1 = target_samples + t * target_tangents
    # target_samples_2 = (
    #     target_samples
    #     + t * target_tangents
    #     - 0.5 * t * t * target_normals * kappa[:, None]
    # )

    # ic(anchor_samples.shape)
    # ps.register_point_cloud("anchor_samples", anchor_samples)
    # ps.register_point_cloud("target_samples", target_samples).add_vector_quantity(
    #     "target_tangents", target_tangents
    # )
    # ps.register_point_cloud("target_samples_1", target_samples_1)
    # ps.register_point_cloud("target_samples_2", target_samples_2)
    # ps.show()

    # exit()

    checkpointer = CheckPointer()

    if args.eval:
        mash = checkpointer.deserialize(mash)
        ps.init()
        ps.register_surface_mesh("Bunny", mesh_tri.vertices, mesh_tri.faces)
        ps_draw_mash_samples(mash, sample_data)
        ps.show()

    else:
        lr = 2e-3
        num_gt_samples = 10000

        chamfer_distance = chamfer_distance_init(
            anchor_size * sample_phi_num * sample_theta_num, num_gt_samples
        )

        optim = optax.adam(lr)
        opt_state = optim.init(mash)

        @jit
        def loss_func(
            mash: Anchor,
            sample_data: AnchorSampleData,
            gt_samples: Array,
            fid: Array,
            w_chamfer1: Float,
            w_chamfer2: Float,
            w_curvature: Float,
        ):
            anchor_samples, anchor_tangents, _ = eval_mash_first_order(
                mash, sample_data
            )

            anchor_samples = anchor_samples.reshape(-1, 3)
            # Index 1 is the one that points outwards
            anchor_tangent = anchor_tangents[..., 1].reshape(-1, 3)

            dist1, idx1, dist2, idx2 = chamfer_distance(anchor_samples, gt_samples)

            target_normals = FN[fid][idx1]
            target_S2s = S2s[fid][idx1]

            target_tangents = vmap(proj_vec)(anchor_tangent, target_normals)
            target_tangents = vmap(normalize)(target_tangents)
            target_tangents = jax.lax.stop_gradient(target_tangents)

            kappa = (
                jnp.einsum("nij,nj->ni", target_S2s, target_tangents) * target_tangents
            ).sum(-1)
            t = 0.005
            target_samples = (
                gt_samples[idx1]
                + t * target_tangents
                - 0.5 * t * t * target_normals * kappa[:, None]
            )
            loss_chamfer = (
                0.01 * vmap(jnp.linalg.norm)(anchor_samples - target_samples).mean()
                + dist1.mean()
            )

            loss = loss_chamfer
            loss_dict = {"loss": loss, "loss_chamfer": loss_chamfer}
            return loss, loss_dict

        @jit
        def make_step(mash: Anchor, opt_state: PyTree, loss_cfg: PyTree):
            grads, loss_dict = grad(loss_func, has_aux=True)(
                mash, sample_data, gt_samples, fid, **loss_cfg
            )
            updates, opt_state = optim.update(grads, opt_state, mash)
            mash = optax.apply_updates(mash, updates)
            return mash, opt_state, loss_dict

        # TODO: Resample for each epoch
        gt_samples, fid = trimesh.sample.sample_surface_even(mesh_tri, num_gt_samples)
        gt_samples = jnp.asarray(gt_samples)

        is_training = False

        traced = make_step.trace(
            mash, opt_state, {"w_chamfer1": 1, "w_chamfer2": 1, "w_curvature": 0}
        )
        lowered = traced.lower()
        compiled = lowered.compile()

        viz_helper = MashVizHelper()
        viz_helper.register_mash(mash, sample_data)

        loss = 0

        def callback():
            nonlocal is_training, mash, opt_state, loss
            _, is_training = psim.Checkbox("Is training", is_training)

            # TODO: Make a chart
            psim.Text(f"Loss: {loss}")

            if is_training:
                mash, opt_state, loss_dict = compiled(
                    mash,
                    opt_state,
                    {"w_chamfer1": 1, "w_chamfer2": 1, "w_curvature": 0},
                )
                loss = loss_dict["loss"]

                viz_helper.update_mash(mash, sample_data)

            if psim.Button("Reset"):
                mash = initialize_anchors_mesh(mesh_tri, anchor_size, init_dist)
                opt_state = optim.init(mash)
                viz_helper.update_mash(mash, sample_data)

        ps.set_user_callback(callback)
        ps.show()
        exit()

        checkpointer.serialize(mash, "bunny-fitting")


if __name__ == "__main__":
    main()
