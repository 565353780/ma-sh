import os
from tqdm import trange

from ma_sh.Model.mash import Mash

if __name__ == "__main__":
    home = os.environ["HOME"]
    mash_folder_path = home + "/chLi/Dataset/TRELLIS/mash_gen/"
    iter_num = 100
    device = "cuda:0"
    cuda_id = "0"
    save_gen_process_folder_path = home + "/chLi/Dataset/TRELLIS/mash_gen_process/"

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id

    mash_filename_list = os.listdir(mash_folder_path)

    for mash_filename in mash_filename_list:
        if not mash_filename.endswith(".npy"):
            continue

        mash_id = mash_filename[:-4]

        mash_file_path = mash_folder_path + mash_filename

        mash = Mash.fromParamsFile(mash_file_path, 10, 10, device=device)
        mask1 = mash.mask_params.clone()
        sh1 = mash.sh_params.clone()
        p1 = mash.positions.clone()
        r1 = mash.ortho_poses.clone()

        mash.randomInit()
        mask0 = mash.mask_params.clone()
        sh0 = mash.sh_params.clone()
        p0 = mash.positions.clone()
        r0 = mash.ortho_poses.clone()

        print("start save gen process for:", mash_id)
        for i in trange(iter_num + 1):
            t = 1.0 * i / iter_num
            t = t**0.1

            maskt = mask1 * t + (1 - t) * mask0
            sht = sh1 * t + (1 - t) * sh0
            pt = p1 * t + (1 - t) * p0
            rt = r1 * t + (1 - t) * r0

            mash.loadParams(maskt, sht, rt, pt)

            save_gen_process_pcd_file_path = (
                save_gen_process_folder_path + mash_id + "/" + str(i) + "_train_pcd.ply"
            )
            mash.saveAsPcdFile(save_gen_process_pcd_file_path)
