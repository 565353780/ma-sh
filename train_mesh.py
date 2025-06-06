from ma_sh.Demo.mesh_trainer import demo as demo_train_mesh


if __name__ == "__main__":
    shape_data_dict = {
        "XiaomiSU7": "/home/chli/chLi/Dataset/XiaomiSU7/Xiaomi_SU7_2024_low_mesh.obj",
        "RobotArm": "/home/chli/chLi/Dataset/RobotArm/Rmk3.obj",
        "Washer": "/home/chli/chLi/Dataset/Washer/BOSCH_WLG.obj",
        "bunny": "/home/chli/chLi/Dataset/Famous/bunny.ply",
        "difficult-0": "/home/chli/chLi/Dataset/vae-eval/manifold/000.obj",
        "difficult-1": "/home/chli/chLi/Dataset/vae-eval/manifold/001.obj",
        "difficult-2": "/home/chli/chLi/Dataset/vae-eval/manifold/002.obj",
        "difficult-3": "/home/chli/chLi/Dataset/vae-eval/manifold/003.obj",
        "difficult-4": "/home/chli/chLi/Dataset/vae-eval/manifold/004.obj",
    }

    save_root_folder_path = "/home/chli/chLi/Results/ma-sh/MeshTrainer/"

    demo_train_mesh(
        shape_data_dict["difficult-0"],
        anchor_num=400,
        mask_degree_max=3,
        sh_degree_max=2,
        save_freq=1,
        save_log_folder_path=save_root_folder_path + "logs/difficult-0/",
        save_result_folder_path=save_root_folder_path + "results/difficult-0/",
    )
