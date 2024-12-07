from torch import profiler

from ma_sh.Demo.trainer import demo as demo_train

if __name__ == "__main__":
    if False:
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA],
            on_trace_ready=profiler.tensorboard_trace_handler('./logs/')
        ) as prof:
            demo_train(400)

        print(prof.key_averages().table(sort_by="cpu_time_total"))
        exit()

    demo_train(400)
    exit()

    demo_train(200)
    demo_train(100)
    demo_train(50)
    demo_train(20)
    demo_train(10)
