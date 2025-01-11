from ma_sh.Demo.adaptive_trainer import demo as demo_train_adaptive

if __name__ == "__main__":
    init_anchor_num_list = [50]
    max_fit_error_list = [1e-3, 1e-2, 1e-1]
    save_freq = 1

    for init_anchor_num in init_anchor_num_list:
        for max_fit_error in max_fit_error_list:
            save_log_folder_path = './logs/anchor-' + str(init_anchor_num) + '_err-' + str(max_fit_error) + '/'
            save_result_folder_path = './output/fit/anchor-' + str(init_anchor_num) + '_err-' + str(max_fit_error) + '/'

            demo_train_adaptive(init_anchor_num,
                                max_fit_error,
                                save_freq,
                                save_log_folder_path,
                                save_result_folder_path)
