import numpy as np
from tqdm import tqdm
from typing import Union


def getNearIdxs(
    point: Union[list, np.ndarray], points: Union[list, np.ndarray]
) -> np.ndarray:
    point = np.array(point, dtype=float)
    points = np.array(points, dtype=float)

    dists = np.linalg.norm(points - point, ord=2, axis=1)
    return dists.argsort()


def getFarIdxs(
    point: Union[list, np.ndarray], points: Union[list, np.ndarray]
) -> np.ndarray:
    return getNearIdxs(point, points)[::-1]


def getNearestPointIdx(
    point: Union[list, np.ndarray], points: Union[list, np.ndarray]
) -> int:
    return getNearIdxs(point, points)[0]


def getFarestPointIdx(
    point: Union[list, np.ndarray], points: Union[list, np.ndarray]
) -> int:
    return getFarIdxs(point, points)[0]


def getAllNearIdxs(points: Union[list, np.ndarray]) -> np.ndarray:
    return np.vstack([getNearIdxs(point, points) for point in tqdm(points)])
