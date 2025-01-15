from time import sleep
from ma_sh.Demo.Convertor.pipeline_convertor import demoToSDFObjaverse, demoToSDFShapeNet

if __name__ == "__main__":
    keep_alive = False

    while True:
        # demoToSDFObjaverse(0.25)
        # demoToSDFObjaverse(0.025)
        # demoToSDFObjaverse(0.0025)

        demoToSDFShapeNet(0.25)
        # demoToSDFShapeNet(0.025)
        # demoToSDFShapeNet(0.0025)

        if not keep_alive:
            break

        sleep(1)
