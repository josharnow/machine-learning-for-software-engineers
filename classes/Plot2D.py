import typing as t
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as fig
if t.TYPE_CHECKING:
    from . import Dataset

class Plot2D(fig.Figure):
    def __init__(self, dataset: 'Dataset', x_label = "", y_label = ""):
        super().__init__()
        self.test = "test"
        self.set_label()

        



    # plt.axis([0, 50, 0, 50])                                 # scale axes (0 to 50)
    # plt.xticks(fontsize=14)                                  # set x axis ticks
    # plt.yticks(fontsize=14)                                  # set y axis ticks
    # plt.xlabel("Reservations", fontsize=14)                  # set x axis label
    # plt.ylabel("Pizzas", fontsize=14)                        # set y axis label
    # X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)  # load data
    # plt.plot(X, Y, "bo")                                     # plot data
    # plt.show()                                               # display chart