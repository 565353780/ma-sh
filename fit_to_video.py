import os

from ma_sh.Method.video import imagesToVideo

if __name__ == '__main__':
    images_folder_path = './output/fit_render/anchor-50_err-0.1/pcd/'
    save_video_file_path = './output/fit_video/anchor-50_err-0_1.mp4'
    video_width = 540
    video_height = 540
    video_fps = 24
    background_color = [255, 255, 255]
    overwrite = False

    image_filename_list = os.listdir(images_folder_path)

    valid_image_filename_list = []

    for image_filename in image_filename_list:
        if image_filename[-4:] not in ['.png', '.jpg']:
            continue

        valid_image_filename_list.append(image_filename)

    valid_image_filename_list.sort(key=lambda x: int(x.split('_')[0]))

    image_file_path_list = [images_folder_path + image_filename for image_filename in valid_image_filename_list]

    imagesToVideo(
        image_file_path_list,
        save_video_file_path,
        video_width,
        video_height,
        video_fps,
        background_color,
        overwrite,
    )
