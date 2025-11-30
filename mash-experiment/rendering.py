import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import jax
from jax import numpy as jnp, vmap, jit, grad
from matplotlib import pyplot as plt

from anchor import (
    Anchor,
    AnchorSampleData,
    eval_sample,
    initialize_anchors_pc,
    unroll_spherical_coords,
    ps_draw_mash_samples,
    eval_mash_vertices,
    eval_mash_tessellation,
)
from mvs_pose import parse_colmap

import drjit as dr
import mitsuba as mi
import math
import trimesh

import polyscope as ps
from icecream import ic


@dr.wrap(source="drjit", target="jax")
@jax.jit
def eval_mash_vertices_mitsuba(mask_params, sh_params_h, pos, rot6d):
    sample_phis, sample_thetas = unroll_spherical_coords(sample_data)
    spherical_samples = jnp.stack([sample_phis, sample_thetas], axis=-1).reshape(-1, 2)
    spherical_samples = jnp.vstack([spherical_samples, jnp.zeros((1, 2))])

    anchor_samples = vmap(
        vmap(eval_sample, in_axes=(0, None, None, None, None, None)),
        in_axes=(None, 0, 0, 0, 0, None),
    )(
        spherical_samples,
        mask_params,
        sh_params_h,
        pos,
        rot6d,
        sample_data,
    )
    return anchor_samples.reshape(-1, 3)


# Modified from: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
def fibonacci_sphere(samples):
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    xyz = np.array(points)
    return xyz


if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb", "llvm_ad_rgb")

    np.random.seed(0)

    num_view_ref = 100
    # Avoid lookup singularities
    origins = 3 * fibonacci_sphere(num_view_ref + 2)[1:-1]
    sensors_ref = []

    for i in range(num_view_ref):
        sensors_ref.append(
            mi.load_dict(
                {
                    "type": "perspective",
                    "to_world": mi.ScalarTransform4f().look_at(
                        origin=list(origins[i]),
                        target=[0.0, 0.0, 0.0],
                        up=[0, 1, 0],
                    ),
                    "fov": 60,
                    "film": {
                        "type": "hdrfilm",
                        "width": 256,
                        "height": 256,
                        "rfilter": {"type": "gaussian"},
                        "sample_border": True,
                    },
                    "sampler": {"type": "independent", "sample_count": 128},
                }
            )
        )

    mesh = mi.load_dict(
        {
            "type": "obj",
            "filename": "assets/bunny.obj",
            "bsdf": {"type": "diffuse"},
        }
    )

    scene_ref = mi.load_dict(
        {
            "type": "scene",
            "integrator": {
                "type": "direct_projective",
                "sppi": 0,
            },
            "emitter": {"type": "constant"},
            # "emitter": {
            #     "type": "envmap",
            #     "filename": "./assets/envmap2.exr",
            # },
            "bunny": mesh,
        }
    )

    ref_save_dir = "output/ref"
    if not os.path.exists(ref_save_dir):
        os.makedirs(ref_save_dir)

    img_ref_list = []
    for i in range(num_view_ref):
        img_ref = mi.render(scene_ref, sensor=sensors_ref[i], spp=256)
        img_ref_list.append(img_ref)
        tag = str(i).zfill(3)
        mi.Bitmap(img_ref).convert(
            mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8
        ).write(f"{ref_save_dir}/{tag}.png")

    # TODO: Streamline MVS
    fov, Ts, pc, pc_n = parse_colmap("assets/colmap")

    num_view = 8
    view_idx = np.random.choice(np.arange(num_view_ref), num_view, replace=False)

    anchor_size = 400
    pc_idx = np.random.choice(np.arange(len(pc)), anchor_size, replace=False)
    pc = pc[pc_idx]
    pc_n = pc_n[pc_idx]

    nz = np.array([0.0, 0.0, -1.0])
    y = np.array([0.0, 1.0, 0.0])

    sensors = []
    for i in view_idx:
        R = Ts[i][:3, :3]
        t = Ts[i][:3, 3]
        sensor = mi.load_dict(
            {
                "type": "perspective",
                "to_world": mi.ScalarTransform4f().look_at(
                    origin=t,
                    target=R @ nz,
                    up=(R @ y - t),
                ),
                "fov": fov,
                "film": {
                    "type": "hdrfilm",
                    "width": 256,
                    "height": 256,
                    "rfilter": {"type": "gaussian"},
                    "sample_border": True,
                },
                "sampler": {"type": "independent", "sample_count": 128},
            }
        )
        sensors.append(sensor)

    sample_phi_num = 40
    sample_theta_num = 30
    init_dist = 5e-2

    sample_data = AnchorSampleData.init(
        sample_phi_num=sample_phi_num, sample_theta_num=sample_theta_num
    )
    mash = initialize_anchors_pc(pc, pc_n, init_dist)

    # Let's try the simple tessellation first
    #   My hypothesis is that rendering gradient won't make extreme deformation such that tessellation distortion appears
    F_tess = eval_mash_tessellation(sample_phi_num, sample_theta_num, anchor_size)
    NF = len(F_tess)
    F_tess = mi.Vector3u(np.asarray(F_tess).T)
    NV = (sample_phi_num * sample_theta_num + 1) * anchor_size

    props = mi.Properties()
    props["mesh_bsdf"] = mi.load_dict(
        {
            "type": "diffuse",
            # "reflectance": {"type": "rgb", "value": [0.3, 0.3, 0.75]},
        }
    )
    mesh = mi.Mesh(
        "anchor",
        vertex_count=NV,
        face_count=NF,
        has_vertex_normals=False,
        has_vertex_texcoords=False,
        props=props,
    )

    mesh_params = mi.traverse(mesh)
    mesh_params["faces"] = dr.ravel(F_tess)
    mesh_params.update()

    scene = mi.load_dict(
        {
            "type": "scene",
            "integrator": {
                "type": "direct_projective",
                "sppi": 0,
            },
            "emitter": {"type": "constant"},
            # "emitter": {
            #     "type": "envmap",
            #     "filename": "./assets/envmap2.exr",
            # },
            "anchor": mesh,
        }
    )

    mash: Anchor = jax.tree_util.tree_map(lambda x: dr.auto.ad.TensorXf(x), mash)

    opt = mi.ad.Adam(lr=3e-2)
    opt["mask_params"] = mash.mask_params
    opt["sh_params_h"] = mash.sh_params_h
    opt["pos"] = mash.pos
    opt["rot6d"] = mash.rot6d

    for it in range(50):
        total_loss = mi.Float(0.0)
        params = mi.traverse(scene)
        V_tess = eval_mash_vertices_mitsuba(
            opt["mask_params"], opt["sh_params_h"], opt["pos"], opt["rot6d"]
        )
        params["anchor.vertex_positions"] = V_tess
        params.update()

        for sensor_idx in range(num_view):
            img = mi.render(scene, params, sensor=sensors[sensor_idx], spp=256)

            if it == 0:
                mi.Bitmap(img).convert(
                    mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8
                ).write(f"output/init_{view_idx[sensor_idx]}.png")

            loss = dr.mean(dr.abs(img - img_ref_list[view_idx[sensor_idx]]))
            total_loss += loss

        dr.backward(total_loss)
        opt.step()
        ic(it, total_loss)

    for sensor_idx in range(num_view_ref):
        img = mi.render(scene, params, sensor=sensors[sensor_idx], spp=256)
        mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8).write(
            f"output/out_{view_idx[sensor_idx]}.png"
        )
