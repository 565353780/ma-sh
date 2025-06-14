from ma_sh.Data.simple_mash import SimpleMash

if __name__ == "__main__":
    mash = SimpleMash(
        anchor_num=3,
        mask_degree_max=2,
        sh_degree_max=1,
        sample_phi_num=3,
        sample_theta_num=2,
    )

    mash.setGradState(True)

    mask_thetas = mash.toMaskThetas()
    print(mask_thetas)

    weighted_sample_phi_theta_mat = mash.toWeightedSamplePhiThetaMat()
    print(weighted_sample_phi_theta_mat)

    mean = weighted_sample_phi_theta_mat.mean()
    mean.backward()

    print(mash.mask_params.grad)
