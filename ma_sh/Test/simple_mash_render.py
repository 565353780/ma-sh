import torch
import open3d as o3d

from ma_sh.Model.simple_mash import SimpleMash
from ma_sh.Test.init_values import initValues

def test():
    anchor_num = 2
    mask_degree_max = 3
    sh_degree_max = 2
    sample_phi_num = 4
    sample_theta_num = 1
    use_inv = True
    idx_dtype = torch.int64
    dtype = torch.float32
    device = "cuda:0"

    mash = SimpleMash(
        anchor_num,
        mask_degree_max,
        sh_degree_max,
        sample_phi_num,
        sample_theta_num,
        use_inv,
        idx_dtype,
        dtype,
        device,
    )

    initValues(mash.mask_params, mash.sh_params, mash.rotate_vectors, mash.positions, 2)

    sample_mesh = mash.toSampleMesh().toO3DMesh()
    #triangles = np.asarray(sample_mesh.triangles)
    #sample_mesh.triangles = o3d.utility.Vector3iVector(triangles[0].reshape(-1, 3))
    sample_ellipses = mash.toSimpleSampleO3DEllipses()

    o3d.visualization.draw_geometries(sample_ellipses + [sample_mesh])
    return True
