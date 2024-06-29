import torch

from ma_sh.Method.render import renderPoints, renderGeometries
from ma_sh.Model.simple_mash import SimpleMash

a = SimpleMash(2, 3, 2, 10, 10, device='cuda')
for i in range(2):
    a.positions[i, 0] = 0.1 * i
for i in range(2):
    a.rotate_vectors[i, 0] = 0.1 * i

points = a.toSamplePoints()
mesh = a.toSampleMesh()
centers, radius, rotations = a.toSimpleSampleSurfels()
circles = a.toSimpleSampleCircles()
print(rotations)
print(torch.bmm(rotations, rotations.transpose(1, 2)))
print(centers.shape)
print(radius.shape)
print(rotations.shape)

renderGeometries(circles)
exit()

mesh.render()

#renderPoints(points.detach().clone().cpu().numpy())
