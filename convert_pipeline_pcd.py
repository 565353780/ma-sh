from time import sleep
from ma_sh.Demo.Convertor.pipeline_convertor import demoToPCDObjaverse, demoToPCDShapeNet

if __name__ == "__main__":
    keep_alive = False

    while True:
        # demoToPCDObjaverse()
        demoToPCDShapeNet()

        if not keep_alive:
            break
        sleep(1)
