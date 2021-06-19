# Author: Solomon Reinman 

import matplotlib.pyplot as plt


def plot_data(pos, neg, p_c, n_c):
    """
    Plot data and centroids.

    Only works for 2-dimensional data.

    Parameters 
    ----------
    pos: { numpy.ndarray } of shape (n_positive_samples, n_features)
        Positive instances
    neg: { numpy.ndarray } of shape (n_negative_samples, n_features)
        Negative instances
    p_c: { numpy.ndarray } of shape (1, n_features)
        Positive centroid
    n_c: { numpy.ndarray } of shape (1, n_features)
        Negative centroid
    """

    fig_width = 5
    fig_height = 3
    panel_width = 2.8 / fig_width
    panel_height = 1.9 / fig_height

    plt.figure(figsize=(fig_width, fig_height))
    panel = plt.axes((0.25, 0.25, panel_width, panel_height))

    panel.set_title("data and calculated centroids")
    panel.set_xlabel("x")
    panel.set_ylabel("y")

    # print(pos)
    # print(neg)

    pos_x = np.hsplit(pos, 2)[0]
    pos_y = np.hsplit(pos, 2)[1]

    neg_x = np.hsplit(neg, 2)[0]
    neg_y = np.hsplit(neg, 2)[1]

    panel.scatter(pos_x, pos_y, s=0.25, c='blue')
    panel.scatter(neg_x, neg_y, s=0.25, c='orange')

    panel.scatter(p_c[0], p_c[1], s=0.5, marker='X', c='darkblue')
    panel.scatter(n_c[0], n_c[1], s=0.5, marker='X', c='red')

    plt.savefig("data.png", dpi=600)

