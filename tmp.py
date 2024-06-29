from ma_sh.Method.render import renderPoints
from ma_sh.Model.simple_mash import SimpleMash

a = SimpleMash(400, 3, 2, 10, 10, device='cuda')
for i in range(10):
    a.positions[i, 0] = 0.1 * i

points = a.toSamplePoints()
mesh = a.toSampleMesh()

mesh.render()

#renderPoints(points.detach().clone().cpu().numpy())
