import torch
import mash_cpp
from ma_sh.Module.timer import Timer


def test():
    timer = Timer()

    sample_point_num = 50000

    timer.reset()
    points = torch.randn([1, 100000, 3], dtype=torch.float32).cuda()
    print("points to cuda", timer.now())

    for _ in range(4):
        timer.reset()
        # FIXME: after run furthest_point_sampling, the next to GPU data ops will be too slow!
        _ = torch.arange(1).cuda()
        print("1st: create new_cuda_data:", timer.now())

        timer.reset()
        _ = torch.arange(10000).cuda()
        print("2nd: create new_cuda_data:", timer.now())

        timer.reset()
        _ = mash_cpp.furthest_point_sampling(points, sample_point_num)
        print("furthest_point_sampling", timer.now())

        print("===============================")

    exit()
    return True
