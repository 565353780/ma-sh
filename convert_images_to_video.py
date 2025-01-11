from time import sleep

from ma_sh.Demo.Convertor.images_to_video import demo as demo_convert_images_to_video

if __name__ == '__main__':
    keep_alive = True

    while True:
        demo_convert_images_to_video()
        if not keep_alive:
            break
        sleep(1)
