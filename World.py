import pandas as pd
import numpy as np


class World:

    def __init__(self):

        self.nRows = 3
        self.nCols = 4
        self.stateObstacles = 5
        self.stateTerminals = [10, 11]
        self.nStates = 12
        self.nActions = 4
