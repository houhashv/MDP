from World import World
import numpy as np


if __name__ == "__main__":

    world = World()
    world.plot()
    world.plot_value([np.random.random() for i in range(12)])

