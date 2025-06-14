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

    print(mash.sample_phis)
    print(mash.sample_thetas)
    print(mash.sample_phi_theta_mat)
    print(mash.sample_phi_theta_mat.shape)

    print(mash.sample_phi_theta_mat[:, :, 1])

    mask_thetas = mash.toMaskThetas()
    print(mask_thetas)

    mean = mask_thetas.mean()
    mean.backward()

    print(mash.mask_params.grad)
