from ma_sh.Method.animation import createLogAnimation

if __name__ == '__main__':
    for anchor_num in [10, 20, 50, 100, 200, 400]:
        print('start create log animation of ', anchor_num, '...')
        createLogAnimation(
            event_folder_path='/home/chli/chLi/Results/ma-sh/output/fit/fixed/bunny/logs/anchor-' + str(anchor_num) + '/',
            tag='Metric/chamfer_distance',
            save_video_file_path='/home/chli/chLi/Results/ma-sh/output/fit/fixed/bunny/video/anchor-' + str(anchor_num) + '_CD.mp4',
            x_label='Step',
            y_label='L1-Chamfer',
            title='MASH Optimization',
            fps=90,
            render=False,
            overwrite=False,
        )
