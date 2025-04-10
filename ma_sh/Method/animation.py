import os
import matplotlib.pyplot as plt
from typing import Union, Tuple
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tensorboard.backend.event_processing import event_accumulator

from ma_sh.Method.path import createFileFolder, removeFile


def extract_scalar(
    event_folder_path: str,
    tag: str,
) -> Union[Tuple[list, list], Tuple[None, None]]:
    if not os.path.exists(event_folder_path):
        print('[ERROR][animation::extract_scalar]')
        print('\t event folder not exist!')
        print('\t event_folder_path:', event_folder_path)
        return None, None

    event_file_path = None

    event_file_name_list = os.listdir(event_folder_path)
    for event_file_name in event_file_name_list:
        if not event_file_name.startswith('event'):
            continue

        event_file_path = event_folder_path + event_file_name
        break

    if event_file_path is None:
        print('[ERROR][animation::extract_scalar]')
        print('\t event file not exist!')
        return None, None

    ea = event_accumulator.EventAccumulator(event_file_path)
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        raise ValueError(f"Tag '{tag}' not found. Available: {ea.Tags()['scalars']}")
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

def createAnimation(
    steps: list,
    values: list,
    save_video_file_path: str,
    x_label: str = 'X',
    y_label: str = 'Y',
    title: str = 'Title',
    fps: int = 30,
    render: bool = False,
    overwrite: bool = False,
) -> bool:
    if len(steps) != len(values):
        print('[ERROR][animation::createAnimation]')
        print('\t steps and values length not matched!')
        return False

    if os.path.exists(save_video_file_path):
        if not overwrite:
            return True

        removeFile(save_video_file_path)

    max_frames = len(values)

    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(0, max(steps))
    ax.set_ylim(0, max(values) * 1.1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    fig.tight_layout()

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        line.set_data(steps[:frame + 1], values[:frame + 1])
        return line,

    ani = FuncAnimation(
        fig, update, frames=max_frames, init_func=init,
        blit=True, repeat=False,
    )

    if render:
        plt.show()

    createFileFolder(save_video_file_path)

    writer = FFMpegWriter(fps=fps, metadata=dict(artist='chLi'), bitrate=1800)
    ani.save(
        save_video_file_path,
        writer=writer,
        dpi=300,
    )

    plt.savefig(save_video_file_path[:-4] + '.png', dpi=300, bbox_inches='tight')

    plt.close()
    return True

def createLogAnimation(
    event_folder_path: str,
    tag: str,
    save_video_file_path: str,
    x_label: str = 'X',
    y_label: str = 'Y',
    title: str = 'Title',
    fps: int = 30,
    render: bool = False,
    overwrite: bool = False,
) -> bool:
    steps, values = extract_scalar(event_folder_path, tag)
    if steps is None or values is None:
        print('[ERROR][animation::createLogAnimation]')
        print('\t extract_scalar failed!')
        return False

    if not createAnimation(
        steps,
        values,
        save_video_file_path,
        x_label,
        y_label,
        title,
        fps,
        render,
        overwrite,
    ):
        print('[ERROR][animation::createLogAnimation]')
        print('\t createAnimation failed!')
        return False

    return True
