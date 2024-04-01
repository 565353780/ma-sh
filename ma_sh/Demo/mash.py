import torch

from ma_sh.Model.mash import Mash


def demo():
    anchor_num = 40
    mask_degree_max = 4
    sh_degree_max = 4
    mask_boundary_sample_num = 18
    sample_polar_num = 4000
    sample_point_scale = 0.4
    use_inv = True
    idx_dtype = torch.int64
    dtype = torch.float64
    device = "cpu"

    if False:
        mash = Mash(
            anchor_num,
            mask_degree_max,
            sh_degree_max,
            mask_boundary_sample_num,
            sample_polar_num,
            sample_point_scale,
            use_inv,
            idx_dtype,
            dtype,
            device,
        )

        for i in range(anchor_num):
            mash.mask_params.data[i, 0] = i + 1.0

        for i in range(anchor_num):
            mash.sh_params.data[i, 0] = i + 1.0

        for i in range(anchor_num):
            mash.rotate_vectors.data[i, 0] = i

        for i in range(anchor_num):
            mash.positions.data[i, 0] = i

        sh_points = mash.toSamplePoints()

        print(sh_points.shape)

        mash.renderSamplePoints()

    if True:
        mash_params_file_path = "/home/chli/Dataset/aro_net/data/shapenet/mash/02691156/92b7d0035cefb816d13ef00338ba8c52_obj.npy"

        mash = Mash.fromParamsFile(mash_params_file_path, 18, 4000, 0.4)
        print(mash_params_file_path)
        print(mash.mask_params[0])
        print(mash.sh_params[0])
        print(mash.rotate_vectors[0])
        print(mash.positions[0])
        exit()

        mash.renderSamplePoints()

        exit()

    return True
