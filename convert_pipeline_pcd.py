from time import sleep
from ma_sh.Demo.Convertor.pipeline_convertor import demoGLB2PCDObjaverse, demoGLB2PCDShapeNet

if __name__ == "__main__":
    keep_alive = False

    while True:
        # demoGLB2PCDObjaverse()
        demoGLB2PCDShapeNet()

        if not keep_alive:
            break
        sleep(1)
