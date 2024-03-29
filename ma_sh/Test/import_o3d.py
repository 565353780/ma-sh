import open3d
import torch


def test():
    a = torch.randn(100)
    print("start calculate 2 * a, a.shape=", a.shape)
    b = 2 * a
    print("finish calculate 2 * a")

    a = torch.randn(1000)
    print("start calculate 2 * a, a.shape=", a.shape)
    b = 2 * a
    print("finish calculate 2 * a")

    a = torch.randn(10000)
    print("start calculate 2 * a, a.shape=", a.shape)
    b = 2 * a
    print("finish calculate 2 * a")

    a = torch.randn(100000)
    print("start calculate 2 * a, a.shape=", a.shape)
    b = 2 * a
    print("finish calculate 2 * a")
    return True
