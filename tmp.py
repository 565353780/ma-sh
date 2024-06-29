from ma_sh.Method.render import renderPoints
from ma_sh.Model.simple_mash import SimpleMash

a = SimpleMash(10, 3, 2, 10, 10)
for i in range(10):
    a.positions[i, 0] = 0.1 * i
points = a.toSamplePoints()

renderPoints(points.detach().clone().cpu().numpy())
