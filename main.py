from World import World
import numpy as np


if __name__ == "__main__":

    world = World()
    world.plot()
    world.plot_value([np.random.random() for i in range(12)])
    world.plot_policy(np.random.randint(1, world.nActions,(world.nStates, 1)))
