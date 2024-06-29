from tqdm import trange
from ma_sh.Method.render import renderPoints
from ma_sh.Model.simple_mash import SimpleMash

a = SimpleMash(400, 3, 2, 10, 10, device='cuda')
for i in range(10):
    a.positions[i, 0] = 0.1 * i

for _ in trange(1000):
    points = a.toSamplePoints()

#renderPoints(points.detach().clone().cpu().numpy())
