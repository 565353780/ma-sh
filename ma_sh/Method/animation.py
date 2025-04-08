import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Union, Tuple
from tensorboard.backend.event_processing import event_accumulator

from ma_sh.Method.path import createFileFolder, removeFile


def extract_scalar(
    event_file_path: str,
    tag: str,
) -> Union[Tuple[list, list], Tuple[None, None]]:
    if not os.path.exists(event_file_path):
        print('[ERROR][animation::extract_scalar]')
        print('\t event file not exist!')
        print('\t event_file_path:', event_file_path)
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
    ax.set_xlabel('Step')
    ax.set_ylabel('Chamfer')
    ax.set_title('MASH Optimization')

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        line.set_data(steps[:frame + 1], values[:frame + 1])
        return line,

    ani = animation.FuncAnimation(
        fig, update, frames=max_frames, init_func=init,
        blit=True, interval=50, repeat=False
    )

    if render:
        plt.show()

    createFileFolder(save_video_file_path)

    ani.save(save_video_file_path, fps=fps, extra_args=['-vcodec', 'libx264'])
    return True

def createLogAnimation(
    event_file_path: str,
    tag: str,
    save_video_file_path: str,
    fps: int = 30,
    render: bool = False,
    overwrite: bool = False,
) -> bool:
    steps, values = extract_scalar(event_file_path, tag)
    if steps is None or values is None:
        print('[ERROR][animation::createLogAnimation]')
        print('\t extract_scalar failed!')
        return False

    if not createAnimation(
        steps,
        values,
        save_video_file_path,
        fps,
        render,
        overwrite,
    ):
        print('[ERROR][animation::createLogAnimation]')
        print('\t createAnimation failed!')
        return False

    return True
