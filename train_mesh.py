import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ma_sh.Demo.mesh_trainer import demo as demo_train_mesh


if __name__ == "__main__":
    home = os.environ["HOME"]
    shape_data_dict = {
        "XiaomiSU7": home + "/chLi/Dataset/XiaomiSU7/Xiaomi_SU7_2024_low_mesh.obj",
        "RobotArm": home + "/chLi/Dataset/RobotArm/Rmk3.obj",
        "Washer": home + "/chLi/Dataset/Washer/BOSCH_WLG.obj",
        "bunny": home + "/chLi/Dataset/Famous/bunny.ply",
        "bunny-v2": home + "/chLi/Dataset/Famous/bunny-v2.ply",
        "difficult-0": home + "/chLi/Dataset/vae-eval/mesh/000.obj",
        "difficult-1": home + "/chLi/Dataset/vae-eval/mesh/001.obj",
        "difficult-2": home + "/chLi/Dataset/vae-eval/mesh/002.obj",
        "difficult-3": home + "/chLi/Dataset/vae-eval/mesh/003.obj",
        "difficult-4": home + "/chLi/Dataset/vae-eval/mesh/004.obj",
    }

    save_root_folder_path = home + "/chLi/Results/ma-sh/MeshTrainer/"

    shape_id = "bunny-v2"

    demo_train_mesh(
        shape_data_dict[shape_id],
        points_per_submesh=1024,
        anchor_num=400,
        mask_degree_max=3,
        sh_degree_max=2,
        sample_phi_num=40,
        sample_theta_num=40,
        device="cpu",
        save_freq=1,
        save_log_folder_path=save_root_folder_path + "logs/" + shape_id + "/",
        save_result_folder_path=save_root_folder_path + "results/" + shape_id + "/",
    )
