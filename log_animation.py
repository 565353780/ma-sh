from ma_sh.Method.animation import createLogAnimation

if __name__ == '__main__':
    event_file_path = '/home/chli/chLi/Results/ma-sh/output/fit/fixed/bunny/logs/anchor-50/events.out.tfevents.1744087733.pop-os.2693090.0'
    tag = 'Metric/chamfer_distance'
    save_video_file_path = '/home/chli/chLi/Results/ma-sh/output/fit/fixed/bunny/video/anchor-50_CD.mp4'
    fps = 30
    render = False
    overwrite = True

    createLogAnimation(
        event_file_path,
        tag,
        save_video_file_path,
        fps,
        render,
        overwrite
    )
