from time import sleep
from ma_sh.Demo.Convertor.pipeline_convertor import demoGLB2SDF

if __name__ == "__main__":
    keep_alive = False

    while True:
        # demoGLB2SDF(0.25)
        demoGLB2SDF(0.025)
        # demoGLB2SDF(0.0025)

        if not keep_alive:
            break

        sleep(1)
