import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.decomposition import PCA


def pca_visualization(semantic_feature: np.ndarray[any, any]) -> np.ndarray[any, any]:
    h, w, c = semantic_feature.shape
    pca = PCA(n_components=3)
    semantic_feature = semantic_feature.flatten().reshape(h * w, c)
    semantic_feature_visualization = pca.fit_transform(semantic_feature)
    semantic_feature_visualization = semantic_feature_visualization.reshape(
        h, w, 3
    ).astype(np.uint8)
    return semantic_feature_visualization


if __name__ == "__main__":
    fig = plt.figure()
    ax = plt.gca()
    ax.set_axis_off()
    semantic_feature = np.load("datasets/temp/semantic_feature.npy")
    ax.imshow(pca_visualization(semantic_feature))
    plt.show()
