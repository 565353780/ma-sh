import numpy as np
import matplotlib.pyplot as plt

def toJetColors(dists: np.ndarray) -> np.ndarray:
    valid_dists = np.clip(dists, 0.0, 1.0)

    labels = 0.0 + 1.0 * valid_dists

    colors = plt.cm.jet(labels)[:, :3]
    return colors

def toWhiteRedColors(dists: np.ndarray) -> np.ndarray:
    valid_dists = np.clip(dists, 0.0, 1.0)

    labels = 0.0 + 1.0 * valid_dists

    colors = plt.cm.Reds(labels)[:, :3]
    return colors

def toRedWhiteColors(dists: np.ndarray) -> np.ndarray:
    valid_dists = np.clip(dists, 0.0, 1.0)

    labels = np.clip(1.0 - valid_dists, 0.0, 1.0)

    colors = plt.cm.Reds(labels)[:, :3]
    return colors

def toBlueWhiteColors(dists: np.ndarray) -> np.ndarray:
    valid_dists = np.clip(dists, 0.0, 1.0)

    labels = np.clip(1.0 - valid_dists, 0.0, 0.6)

    colors = plt.cm.Blues(labels)[:, :3]
    return colors

def toCoolColors(dists: np.ndarray) -> np.ndarray:
    valid_dists = np.clip(dists, 0.0, 1.0)

    labels = 0.2 + 0.6 * valid_dists

    colors = plt.cm.cool(labels)[:, :3]
    return colors

def toSummerColors(dists: np.ndarray) -> np.ndarray:
    valid_dists = np.clip(dists, 0.0, 1.0)

    labels = 0.0 + 1.0 * valid_dists

    colors = plt.cm.summer(labels)[:, :3]
    return colors

def toTerrainColors(dists: np.ndarray) -> np.ndarray:
    valid_dists = np.clip(dists, 0.0, 1.0)

    labels = 0.0 + 1.0 * valid_dists

    colors = plt.cm.terrain(labels)[:, :3]
    return colors

def toRainbowColors(dists: np.ndarray) -> np.ndarray:
    valid_dists = np.clip(dists, 0.0, 1.0)

    labels = 0.0 + 1.0 * valid_dists

    colors = plt.cm.rainbow(labels)[:, :3]
    return colors


def getJetColorsFromDists(dists: np.ndarray) -> np.ndarray:
    return toJetColors(dists)
    return toWhiteRedColors(dists)
    return toRedWhiteColors(dists)
    return toRainbowColors(dists)
    return toTerrainColors(dists)
    return toSummerColors(dists)
    return toCoolColors(dists)
    return toBlueWhiteColors(dists)
