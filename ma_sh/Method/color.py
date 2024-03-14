import numpy as np
import matplotlib.pyplot as plt


def getJetColorsFromDists(dists: np.ndarray) -> np.ndarray:
    return plt.cm.jet(np.clip(dists, -1.0, 1.0))[:, :3]
