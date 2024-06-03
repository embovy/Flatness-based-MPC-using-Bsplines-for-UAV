import numpy as np


class Space:
    def __init__(self, low, high) -> None:
        self.low = np.array(low)
        self.high = np.array(high)

        self.shape = self.low.shape

    def __repr__(self):
        return f"space low: {self.low} \nspace high: {self.high}"

    # def shape(self):
    #     return self.low.shape
