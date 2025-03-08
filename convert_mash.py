from time import sleep

from ma_sh.Demo.Convertor.mash import demo as demo_convert_mash


if __name__ == "__main__":
    keep_alive = False

    while True:
        demo_convert_mash()

        if not keep_alive:
            break

        sleep(1)
