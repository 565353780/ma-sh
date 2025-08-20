import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ma_sh.Demo.pcd_trainer import demo as demo_train_pcd


if __name__ == "__main__":
    home = os.environ["HOME"]
    shape_data_dict = {
        "BitAZ": home + "/chLi/Dataset/BitAZ/pcd/BitAZ.ply",
    }
    shape_id = "BitAZ"

    pcd_file_path = shape_data_dict[shape_id]
    save_root_folder_path = home + "/chLi/Results/ma-sh/MeshTrainer/" + shape_id + "/"

    demo_train_pcd(
        pcd_file_path,
        anchor_num=400,
        mask_degree_max=3,
        sh_degree_max=2,
        save_freq=-1,
        save_log_folder_path=save_root_folder_path + "logs/" + shape_id + "/",
        save_result_folder_path=save_root_folder_path + "results/" + shape_id + "/",
    )
