import numpy as np
import jax
from jax import numpy as jnp, vmap, jit, grad
from jaxtyping import Float, Array
from anchor import (
    Anchor,
    AnchorSampleData,
    eval_anchor_second_order,
    eval_anchor_tessellation,
    eval_anchor_tessellation_double_covering,
    eval_anchor_tip_sample,
)
from common import w0, mean_curvature
from sh_rotation import rotvec_to_R3_expm, rotvec_to_R5_expm

import polyscope as ps
import polyscope.imgui as psim
from icecream import ic


def main():
    ps.init()
    ps.set_ground_plane_height(-0.5)

    pos = jnp.array([0.0, 0.0, 0.0])
    rot6d = jnp.array([0.0, 0.0, -1.0, 1.0, 0.0, 0.0])
    init_dist = 5e-2
    sample_phi_num = 40
    sample_theta_num = 30

    anchor = Anchor.init(pos, rot6d, init_dist)
    sample_data = AnchorSampleData.init(
        sample_phi_num=sample_phi_num, sample_theta_num=sample_theta_num
    )

    ps.register_point_cloud("pos", pos[None, :])
    anchor_samples, anchor_tangents, anchor_normals, anchor_shape_operators = (
        eval_anchor_second_order(anchor, sample_data)
    )
    anchor_curvatures = vmap(mean_curvature)(anchor_shape_operators)

    samples_viz = ps.register_point_cloud("anchor_samples", anchor_samples)
    samples_viz.add_vector_quantity("anchor_normals", anchor_normals, enabled=True)
    samples_viz.add_vector_quantity(
        "anchor_tangents_0", anchor_tangents[..., 0], enabled=True
    )
    samples_viz.add_vector_quantity(
        "anchor_tangents_1", anchor_tangents[..., 1], enabled=True
    )
    samples_viz.add_scalar_quantity(
        "anchor_curvatures",
        anchor_curvatures,
        vminmax=[0, 20],
        enabled=True,
    )

    tip_sample = eval_anchor_tip_sample(anchor, sample_data)
    anchor_tess_faces = eval_anchor_tessellation(sample_phi_num, sample_theta_num)
    anchor_tess_faces_dc = eval_anchor_tessellation_double_covering(
        sample_phi_num, sample_theta_num
    )
    tessellation_viz = ps.register_surface_mesh(
        "anchor_tessellation",
        jnp.vstack([anchor_samples, tip_sample[None, :]]),
        anchor_tess_faces,
    )
    tessellation_dc_viz = ps.register_surface_mesh(
        "anchor_tessellation_dc",
        jnp.vstack([anchor_samples, tip_sample[None, :]]),
        anchor_tess_faces_dc,
        enabled=False
    )

    def update_samples(anchor: Anchor, sample_data, rotvec):
        nonlocal samples_viz, tessellation_viz, tessellation_dc_viz

        R3 = rotvec_to_R3_expm(rotvec)
        R5 = rotvec_to_R5_expm(rotvec)
        sh_params_h = anchor.sh_params_h
        sh_params_h = sh_params_h.at[1:4].set(R3 @ sh_params_h[1:4])
        sh_params_h = sh_params_h.at[4:].set(R5 @ sh_params_h[4:])

        anchor_copy = Anchor(anchor.mask_params, sh_params_h, anchor.pos, anchor.rot6d)
        anchor_samples, anchor_tangents, anchor_normals, anchor_shape_operators = (
            eval_anchor_second_order(
                anchor_copy,
                sample_data,
            )
        )
        samples_viz.update_point_positions(anchor_samples)
        samples_viz.add_vector_quantity("anchor_normals", anchor_normals)
        samples_viz.add_vector_quantity("anchor_tangents_0", anchor_tangents[..., 0])
        samples_viz.add_vector_quantity("anchor_tangents_1", anchor_tangents[..., 1])
        anchor_curvatures = vmap(mean_curvature)(anchor_shape_operators)
        samples_viz.add_scalar_quantity(
            "anchor_curvatures",
            anchor_curvatures,
            vminmax=[0, 20],
            enabled=True,
        )
        tessellation_viz.update_vertex_positions(
            jnp.vstack(
                [
                    anchor_samples,
                    eval_anchor_tip_sample(anchor_copy, sample_data)[None, :],
                ]
            )
        )
        tessellation_dc_viz.update_vertex_positions(
            jnp.vstack(
                [
                    anchor_samples,
                    eval_anchor_tip_sample(anchor_copy, sample_data)[None, :],
                ]
            )
        )

    spiral = False
    height_map = False
    inv_trans = True

    mask_params_0 = anchor.mask_params[0]
    mask_params_cos = anchor.mask_params[1::2]
    mask_params_sin = anchor.mask_params[2::2]

    sh_params_h_0 = anchor.sh_params_h[:3]
    sh_params_h_1 = anchor.sh_params_h[3:6]
    sh_params_h_2 = anchor.sh_params_h[6:]

    rotvec = jnp.zeros((3,))

    def callback():
        nonlocal \
            spiral, \
            height_map, \
            anchor, \
            sample_data, \
            mask_params_0, \
            mask_params_cos, \
            mask_params_sin, \
            sh_params_h_0, \
            sh_params_h_1, \
            sh_params_h_2, \
            rotvec, \
            inv_trans

        if psim.TreeNode("SH param"):
            changed_h0, sh_params_h_0 = psim.DragFloat3(
                "h0", sh_params_h_0, 0.01, -5.0, 5.0
            )
            changed_h1, sh_params_h_1 = psim.DragFloat3(
                "h1", sh_params_h_1, 0.01, -2.0, 2.0
            )
            changed_h2, sh_params_h_2 = psim.DragFloat3(
                "h2", sh_params_h_2, 0.01, -2.0, 2.0
            )

            if changed_h0 or changed_h1 or changed_h2:
                anchor.sh_params_h = anchor.sh_params_h.at[:3].set(sh_params_h_0)
                anchor.sh_params_h = anchor.sh_params_h.at[3:6].set(sh_params_h_1)
                anchor.sh_params_h = anchor.sh_params_h.at[6:].set(sh_params_h_2)
                update_samples(anchor, sample_data, rotvec)

            if psim.Button("Reset SH"):
                anchor.sh_params_h = jnp.zeros_like(anchor.sh_params_h)
                anchor.sh_params_h = anchor.sh_params_h.at[0].set(init_dist / w0)
                sh_params_h_0 = anchor.sh_params_h[:3]
                sh_params_h_1 = anchor.sh_params_h[3:6]
                sh_params_h_2 = anchor.sh_params_h[6:]
                update_samples(anchor, sample_data, rotvec)

            psim.Text("SH rotation")
            changed_rotvec, rotvec = psim.DragFloat3(
                "rotvec", rotvec, 0.01, -2 * jnp.pi, 2 * jnp.pi
            )
            if changed_rotvec:
                update_samples(anchor, sample_data, rotvec)

            if psim.Button("Reset rotvec"):
                rotvec = jnp.zeros((3,))
                update_samples(anchor, sample_data, rotvec)

            psim.TreePop()

        if psim.TreeNode("Mask param"):
            changed_0, mask_params_0 = psim.DragFloat(
                "0", mask_params_0, 0.01, -5.0, 5.0
            )
            changed_cos, mask_params_cos = psim.DragFloat3(
                "cos", mask_params_cos, 0.01, -5.0, 5.0
            )
            changed_sin, mask_params_sin = psim.DragFloat3(
                "sin", mask_params_sin, 0.01, -5.0, 5.0
            )
            if changed_0 or changed_cos or changed_sin:
                anchor.mask_params = anchor.mask_params.at[0].set(mask_params_0)
                anchor.mask_params = anchor.mask_params.at[1::2].set(mask_params_cos)
                anchor.mask_params = anchor.mask_params.at[2::2].set(mask_params_sin)
                update_samples(anchor, sample_data, rotvec)

            if psim.Button("Reset"):
                anchor.mask_params = jnp.zeros_like(anchor.mask_params)
                mask_params_0 = anchor.mask_params[0]
                mask_params_cos = anchor.mask_params[1::2]
                mask_params_sin = anchor.mask_params[2::2]
                update_samples(anchor, sample_data, rotvec)

            psim.TreePop()

        if psim.TreeNode("Anchor config"):
            changed, height_map = psim.Checkbox("Use Height Map", height_map)
            if changed:
                sample_data.height_map = height_map
                update_samples(anchor, sample_data, rotvec)
            changed, inv_trans = psim.Checkbox("Use Inverse Transform", inv_trans)
            if changed:
                sample_data.inv_trans = inv_trans
                update_samples(anchor, sample_data, rotvec)
            psim.TreePop()

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
