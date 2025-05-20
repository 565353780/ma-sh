from time import sleep

from ma_sh.Demo.Convertor.mesh_to_mash import demo as demo_convert_mesh_to_mash


if __name__ == "__main__":
    keep_alive = False

    while True:
        demo_convert_mesh_to_mash(
            4000, "vae-eval/manifold/", "vae-eval/manifold_mash-4000/"
        )

        if not keep_alive:
            break

        sleep(1)
