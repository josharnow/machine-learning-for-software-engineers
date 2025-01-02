import typing as t
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
if t.TYPE_CHECKING:
    from . import Dataset

def plot_2d_data(x: np.ndarray, y: np.ndarray, x_label: str = "", y_label: str = "", x_font_size = 14, y_font_size = 14, title = "", other_data = []) -> None:
    sns.set_theme()
    # scale axes (0 to 50)
    plt.axis([0, 50, 0, 50])
    # set x axis ticks
    plt.xticks(fontsize=x_font_size)
    # set y axis ticks
    plt.yticks(fontsize=y_font_size)
    # set x axis label
    plt.xlabel(x_label, fontsize=x_font_size)
    # set y axis label
    plt.ylabel(y_label, fontsize=y_font_size)
    # set title
    plt.title(title)

    # plot data
    plt.plot(x, y, "bo")

    if other_data:
        plt.plot(other_data[0], other_data[1])


    # plt.plot(x+1, y-1)
    # display chart
    plt.show()

def plot_3d_data(**kwargs) -> None:
    # TODO - Feed data
    # Import the dataset from args
    x1, x2, x3, y = np.loadtxt("data/pizza_3_vars.txt", skiprows=1, unpack=True)

    # These weights came out of the training phase
    w = np.array([-3.98230894, 0.37333539, 1.69202346])



    # Plot the axes
    sns.set_theme(rc={"axes.facecolor": "white", "figure.facecolor": "white"})
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel(kwargs.get('x_label', ''), labelpad=kwargs.get('x_label_pad'), fontsize=kwargs.get('x_font_size'))
    ax.set_ylabel(kwargs.get('y_label', ''), labelpad=kwargs.get('y_label_pad'), fontsize=kwargs.get('y_font_size'))
    ax.set_zlabel(kwargs.get('z_label', ''), labelpad=kwargs.get('z_label_pad'), fontsize=kwargs.get('z_font_size'))

    # Plot the data points
    ax.scatter3D(x1, x2, y, color='b')


    # Plot the plane
    MARGIN = 10
    edges_x = [np.min(x1) - MARGIN, np.max(x1) + MARGIN]
    edges_y = [np.min(x2) - MARGIN, np.max(x2) + MARGIN]
    xs, ys = np.meshgrid(edges_x, edges_y)
    zs = np.array([w[0] + x * w[1] + y * w[2] for x, y in
                zip(np.ravel(xs), np.ravel(ys))])
    ax.plot_surface(xs, ys, zs.reshape((2, 2)), alpha=0.2)

    print(x3)
    plt.show()


def plot_vectors(vectors: list[np.ndarray], colors):
    """
    Plot one or more vectors in a 2D plane, specifying a color for each.

    Arguments
    ---------
    vectors: list of lists or of arrays
        Coordinates of the vectors to plot. For example, [[1, 3], [2, 2]]
        contains two vectors to plot, [1, 3] and [2, 2].
    colors: list
        Colors of the vectors. For instance: ['red', 'blue'] will display the
        first vector in red and the second in blue.

    Example
    -------
    plot_vectors([[1, 3], [2, 2]], ['red', 'blue'])
    plt.xlim(-1, 4)
    plt.ylim(-1, 4)
    """
    plt.figure()
    test = np.array([323])
    plt.axvline(x=0, color='lightgray')
    plt.axhline(y=0, color='lightgray')

    for i in range(len(vectors)):
        x = np.concatenate([[0,0],vectors[i]])
        plt.quiver([x[0]], [x[1]], [x[2]], [x[3]],
                angles='xy', scale_units='xy', scale=1, color=colors[i],)