import jax
from jax import numpy as jnp, jit, vmap, grad
from jaxtyping import Float, Array, Int

import torch
from typing import List

import triton
import triton.language as tl

from wrap_torch2jax import torch2jax, torch2jax_with_vjp

from icecream import ic


@triton.jit
def nm_dist_kernel(
    xyz1_ptr,
    xyz2_ptr,
    dists_ptr,
    indices_ptr,
    N,
    M,
    xyz1_stride_n,
    xyz1_stride_d,
    xyz2_stride_m,
    xyz2_stride_d,
    dist_stride_n,
    indices_stride_n,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    # More L2 cache friendly launch
    num_pid_n = tl.num_programs(axis=0)
    num_pid_m = tl.num_programs(axis=1)
    pid_n, pid_m = tl.swizzle2d(pid_n, pid_m, num_pid_n, num_pid_m, GROUP_SIZE)

    base_n = pid_n * BLOCK_SIZE_N

    batch_base_n = base_n + tl.arange(0, BLOCK_SIZE_N)
    batch_n_mask = batch_base_n < N
    xyz1_x = tl.load(
        xyz1_ptr + batch_base_n * xyz1_stride_n, mask=batch_n_mask, other=100
    )
    xyz1_y = tl.load(
        xyz1_ptr + batch_base_n * xyz1_stride_n + xyz1_stride_d,
        mask=batch_n_mask,
        other=100,
    )
    xyz1_z = tl.load(
        xyz1_ptr + batch_base_n * xyz1_stride_n + 2 * xyz1_stride_d,
        mask=batch_n_mask,
        other=100,
    )

    cur_best_d = tl.full((BLOCK_SIZE_N,), float("inf"), tl.float32)
    cur_best_idx = tl.zeros((BLOCK_SIZE_N,), tl.int32)
    for base_m in tl.range(0, M, BLOCK_SIZE_M):
        batch_base_m = base_m + tl.arange(0, BLOCK_SIZE_M)
        batch_m_mask = batch_base_m < M
        xyz2_x = tl.load(
            xyz2_ptr + batch_base_m * xyz2_stride_m, mask=batch_m_mask, other=-100
        )
        xyz2_y = tl.load(
            xyz2_ptr + batch_base_m * xyz2_stride_m + xyz2_stride_d,
            mask=batch_m_mask,
            other=-100,
        )
        xyz2_z = tl.load(
            xyz2_ptr + batch_base_m * xyz2_stride_m + 2 * xyz2_stride_d,
            mask=batch_m_mask,
            other=-100,
        )

        x2 = xyz1_x[:, None] - xyz2_x[None, :]
        y2 = xyz1_y[:, None] - xyz2_y[None, :]
        z2 = xyz1_z[:, None] - xyz2_z[None, :]
        d = x2 * x2 + y2 * y2 + z2 * z2

        best_d, best_idx = tl.min(d, axis=1, return_indices=True)
        best_idx += base_m

        mask = best_d < cur_best_d
        cur_best_d = tl.where(mask, best_d, cur_best_d)
        cur_best_idx = tl.where(mask, best_idx, cur_best_idx)

    tl.store(dists_ptr + batch_base_n * dist_stride_n, cur_best_d, mask=batch_n_mask)
    tl.store(
        indices_ptr + batch_base_n * indices_stride_n, cur_best_idx, mask=batch_n_mask
    )


def nm_dist(xyz1: torch.Tensor, xyz2: torch.Tensor):
    assert xyz1.shape[-1] == xyz2.shape[-1], "Incompatible dimensions"
    assert xyz1.is_contiguous(), "Matrix xyz1 must be contiguous"
    assert xyz2.is_contiguous(), "Matrix xyz2 must be contiguous"

    N, D = xyz1.shape
    M, D = xyz2.shape

    dists = torch.zeros((N,), device=xyz1.device, dtype=xyz1.dtype)
    indices = torch.zeros((N,), device=xyz1.device, dtype=torch.int32)

    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE_N"]), 1)

    configs = {
        "BLOCK_SIZE_N": 16,
        "BLOCK_SIZE_M": 512,
        "GROUP_SIZE": 16,
        "num_warps": 2,
        "num_stages": 3,
    }

    nm_dist_kernel[grid](
        xyz1,
        xyz2,
        dists,
        indices,
        N,
        M,
        xyz1.stride(0),
        xyz1.stride(1),
        xyz2.stride(0),
        xyz2.stride(1),
        dists.stride(0),
        indices.stride(0),
        **configs,
    )

    return dists, indices


def nm_dist_init(n: Int, m: Int):
    nm_dist_jax = torch2jax(
        nm_dist,
        jax.ShapeDtypeStruct((n, 3), jnp.float32),
        jax.ShapeDtypeStruct((m, 3), jnp.float32),
        output_shapes=(
            jax.ShapeDtypeStruct((n,), jnp.float32),
            jax.ShapeDtypeStruct((n,), jnp.int32),
        ),
    )

    @jax.custom_vjp
    def nm_dist_impl(xyz1, xyz2):
        dist1, idx1 = nm_dist_jax(xyz1, xyz2)
        return (dist1, idx1)

    @jit
    def nm_dist_fwd(xyz1: jax.Array, xyz2: jax.Array):
        dist1, idx1 = nm_dist_jax(xyz1, xyz2)
        return (dist1, idx1), (None, None)

    @jit
    def nm_dist_bwd(res, dz):
        return (None, None)

    nm_dist_impl.defvjp(nm_dist_fwd, nm_dist_bwd)
    return jit(nm_dist_impl)


def _chamfer_distance_torch(
    xyz1: torch.Tensor, xyz2: torch.Tensor
) -> List[torch.Tensor]:
    dist1, idx1 = nm_dist(xyz1, xyz2)
    dist2, idx2 = nm_dist(xyz2, xyz1)
    return dist1, idx1, dist2, idx2


def chamfer_distance_init(n: Int, m: Int):
    chamfer_distance_jax = torch2jax(
        _chamfer_distance_torch,
        jax.ShapeDtypeStruct((n, 3), jnp.float32),
        jax.ShapeDtypeStruct((m, 3), jnp.float32),
        output_shapes=(
            jax.ShapeDtypeStruct((n,), jnp.float32),
            jax.ShapeDtypeStruct((n,), jnp.int32),
            jax.ShapeDtypeStruct((m,), jnp.float32),
            jax.ShapeDtypeStruct((m,), jnp.int32),
        ),
    )

    @jax.custom_vjp
    def chamfer_distance(xyz1, xyz2):
        dist1, idx1, dist2, idx2 = chamfer_distance_jax(xyz1, xyz2)
        return (dist1, idx1, dist2, idx2)

    @jit
    def chamfer_distance_fwd(xyz1: jax.Array, xyz2: jax.Array):
        dist1, idx1, dist2, idx2 = chamfer_distance_jax(xyz1, xyz2)
        return (dist1, idx1, dist2, idx2), (xyz1, idx1, xyz2, idx2)

    @jit
    def chamfer_distance_bwd(res, dz):
        xyz1, idx1, xyz2, idx2 = res
        dz1, _, dz2, _ = dz

        d_dist1 = dz1[:, None] * 2 * (xyz1 - xyz2[idx1])
        d_dist2 = dz2[:, None] * 2 * (xyz2 - xyz1[idx2])

        dxyz1 = d_dist1.at[idx2, :].add(-d_dist2)
        dxyz2 = d_dist2.at[idx1, :].add(-d_dist1)

        return (dxyz1, dxyz2)

    chamfer_distance.defvjp(chamfer_distance_fwd, chamfer_distance_bwd)
    return jit(chamfer_distance)


if __name__ == "__main__":
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    xyz1 = jax.random.normal(key1, (1600, 3))
    xyz2 = jax.random.normal(key2, (1000, 3))

    chamfer_distance_jax = chamfer_distance_init(xyz1, xyz2)

    dist1, idx1, dist2, idx2 = chamfer_distance_jax(xyz1, xyz2)

    ic(dist1)
    ic(idx1)

    def loss_func(xyz1, xyz2):
        dist1, _, dist2, _ = chamfer_distance_jax(xyz1, xyz2)
        return 0.5 * (dist1.mean() + dist2.mean())

    ic(grad(loss_func)(xyz1, xyz2).shape)
