from ma_sh.Method.animation import createLogAnimation

if __name__ == '__main__':
    event_file_path = '/home/chli/chLi/Results/ma-sh/output/fit/fixed/bunny/logs/anchor-50/events.out.tfevents.1744087733.pop-os.2693090.0'
    tag = 'Metric/chamfer_distance'
    save_video_file_path = '/home/chli/chLi/Results/ma-sh/output/fit/fixed/bunny/video/anchor-50_CD.mp4'
    x_label = 'Step'
    y_label = 'L1-Chamfer'
    title = 'MASH Optimization'
    fps = 30
    render = False
    overwrite = True

    createLogAnimation(
        event_file_path,
        tag,
        save_video_file_path,
        x_label,
        y_label,
        title,
        fps,
        render,
        overwrite
    )
