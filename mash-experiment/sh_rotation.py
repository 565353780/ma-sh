import jax
from jax import jit, numpy as jnp
from icecream import ic

Lx_1 = jnp.array(
    [
        [0, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ]
)

Ly_1 = jnp.array(
    [
        [0, 0, 1],
        [0, 0, 0],
        [-1, 0, 0],
    ]
)

Lz_1 = jnp.array(
    [
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ]
)


Lx_2 = jnp.array(
    [
        [0, 0, 0, -1, 0],
        [0, 0, -jnp.sqrt(3 / 2), 0, -1],
        [0, jnp.sqrt(3 / 2), 0, -jnp.sqrt(3 / 2), 0],
        [1, 0, jnp.sqrt(3 / 2), 0, 0],
        [0, 1, 0, 0, 0],
    ]
)

Ly_2 = jnp.array(
    [
        [0, 1, 0, 0, 0],
        [-1, 0, jnp.sqrt(3 / 2), 0, 0],
        [0, -jnp.sqrt(3 / 2), 0, -jnp.sqrt(3 / 2), 0],
        [0, 0, jnp.sqrt(3 / 2), 0, -1],
        [0, 0, 0, 1, 0],
    ]
)

Lz_2 = jnp.array(
    [
        [0, 0, 0, 0, 2],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0],
        [-2, 0, 0, 0, 0],
    ]
)

Lx_3 = jnp.array(
    [
        [0, 0, 0, 0, 0, -jnp.sqrt(3 / 2), 0],
        [0, 0, 0, 0, -jnp.sqrt(5 / 2), 0, -jnp.sqrt(3 / 2)],
        [0, 0, 0, -jnp.sqrt(3), 0, -jnp.sqrt(5 / 2), 0],
        [0, 0, jnp.sqrt(3), 0, -jnp.sqrt(3), 0, 0],
        [0, jnp.sqrt(5 / 2), 0, jnp.sqrt(3), 0, 0, 0],
        [jnp.sqrt(3 / 2), 0, jnp.sqrt(5 / 2), 0, 0, 0, 0],
        [0, jnp.sqrt(3 / 2), 0, 0, 0, 0, 0],
    ]
)

Ly_3 = jnp.array(
    [
        [0, jnp.sqrt(3 / 2), 0, 0, 0, 0, 0],
        [-jnp.sqrt(3 / 2), 0, jnp.sqrt(5 / 2), 0, 0, 0, 0],
        [0, -jnp.sqrt(5 / 2), 0, jnp.sqrt(3), 0, 0, 0],
        [0, 0, -jnp.sqrt(3), 0, -jnp.sqrt(3), 0, 0],
        [0, 0, 0, jnp.sqrt(3), 0, -jnp.sqrt(5 / 2), 0],
        [0, 0, 0, 0, jnp.sqrt(5 / 2), 0, -jnp.sqrt(3 / 2)],
        [0, 0, 0, 0, 0, jnp.sqrt(3 / 2), 0],
    ]
)

Lz_3 = jnp.array(
    [
        [0, 0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0],
        [0, -2, 0, 0, 0, 0, 0],
        [-3, 0, 0, 0, 0, 0, 0],
    ]
)

Lx_4 = jnp.array(
    [
        [0, 0, 0, 0, 0, 0, 0, -jnp.sqrt(2), 0],
        [0, 0, 0, 0, 0, 0, -jnp.sqrt(7 / 2), 0, -jnp.sqrt(2)],
        [0, 0, 0, 0, 0, -3 / jnp.sqrt(2), 0, -jnp.sqrt(7 / 2), 0],
        [0, 0, 0, 0, -jnp.sqrt(5), 0, -3 / jnp.sqrt(2), 0, 0],
        [0, 0, 0, jnp.sqrt(5), 0, -jnp.sqrt(5), 0, 0, 0],
        [0, 0, 3 / jnp.sqrt(2), 0, jnp.sqrt(5), 0, 0, 0, 0],
        [0, jnp.sqrt(7 / 2), 0, 3 / jnp.sqrt(2), 0, 0, 0, 0, 0],
        [jnp.sqrt(2), 0, jnp.sqrt(7 / 2), 0, 0, 0, 0, 0, 0],
        [0, jnp.sqrt(2), 0, 0, 0, 0, 0, 0, 0],
    ]
)


Ly_4 = jnp.array(
    [
        [0, jnp.sqrt(2), 0, 0, 0, 0, 0, 0, 0],
        [-jnp.sqrt(2), 0, jnp.sqrt(7 / 2), 0, 0, 0, 0, 0, 0],
        [0, -jnp.sqrt(7 / 2), 0, 3 / jnp.sqrt(2), 0, 0, 0, 0, 0],
        [0, 0, -3 / jnp.sqrt(2), 0, jnp.sqrt(5), 0, 0, 0, 0],
        [0, 0, 0, -jnp.sqrt(5), 0, -jnp.sqrt(5), 0, 0, 0],
        [0, 0, 0, 0, jnp.sqrt(5), 0, -3 / jnp.sqrt(2), 0, 0],
        [0, 0, 0, 0, 0, 3 / jnp.sqrt(2), 0, -jnp.sqrt(7 / 2), 0],
        [0, 0, 0, 0, 0, 0, jnp.sqrt(7 / 2), 0, -jnp.sqrt(2)],
        [0, 0, 0, 0, 0, 0, 0, jnp.sqrt(2), 0],
    ]
)

Lz_4 = jnp.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 4.0],
        [0, 0, 0, 0, 0, 0, 0, 3.0, 0],
        [0, 0, 0, 0, 0, 0, 2.0, 0, 0],
        [0, 0, 0, 0, 0, 1.0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1.0, 0, 0, 0, 0, 0],
        [0, 0, -2.0, 0, 0, 0, 0, 0, 0],
        [0, -3.0, 0, 0, 0, 0, 0, 0, 0],
        [-4.0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)


@jit
def rotvec_to_R3_expm(rotvec):
    A = rotvec[0] * Lx_1 + rotvec[1] * Ly_1 + rotvec[2] * Lz_1
    return jax.scipy.linalg.expm(A)


@jit
def rotvec_to_R5_expm(rotvec):
    A = rotvec[0] * Lx_2 + rotvec[1] * Ly_2 + rotvec[2] * Lz_2
    return jax.scipy.linalg.expm(A)


@jit
def rotvec_to_R7_expm(rotvec):
    A = rotvec[0] * Lx_3 + rotvec[1] * Ly_3 + rotvec[2] * Lz_3
    return jax.scipy.linalg.expm(A)


@jit
def rotvec_to_R9_expm(rotvec):
    A = rotvec[0] * Lx_4 + rotvec[1] * Ly_4 + rotvec[2] * Lz_4
    return jax.scipy.linalg.expm(A)


if __name__ == "__main__":
    jnp.set_printoptions(linewidth=140)

    @jit
    def lie_bracket(Li, Lj):
        return Li @ Lj - Lj @ Li

    def check_lie_bracket(Lx, Ly, Lz):
        e0 = jnp.linalg.norm(lie_bracket(Lx, Ly) - Lz)
        e1 = jnp.linalg.norm(lie_bracket(Ly, Lz) - Lx)
        e2 = jnp.linalg.norm(lie_bracket(Lz, Lx) - Ly)
        ic(e0, e1, e2)

    check_lie_bracket(Lx_1, Ly_1, Lz_1)
    check_lie_bracket(Lx_2, Ly_2, Lz_2)
    check_lie_bracket(Lx_3, Ly_3, Lz_3)
    check_lie_bracket(Lx_4, Ly_4, Lz_4)
