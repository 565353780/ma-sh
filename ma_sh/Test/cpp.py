import torch
import mash_cpp

def test():
    a = 1
    b = 2
    c = mash_cpp.add(a, b)

    assert c == a + b
    return True
