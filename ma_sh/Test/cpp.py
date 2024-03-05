import sys
sys.path.append('./build')

import mash_cpp

def test():
    a = 1
    b = 2
    c = mash_cpp.add(a, b)
    print(c)
    return True
