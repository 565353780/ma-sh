from time import sleep
from ma_sh.Demo.Convertor.pipeline_convertor import demoToPCDObjaverse, demoToPCDShapeNet, demoToPCDThingi10K

if __name__ == "__main__":
    keep_alive = False

    while True:
        # demoToPCDObjaverse()
        # demoToPCDShapeNet()
        demoToPCDThingi10K()

        if not keep_alive:
            break
        sleep(1)
