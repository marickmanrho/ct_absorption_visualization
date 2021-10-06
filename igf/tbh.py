# Green function for the tight binding hamiltonian (TBH).
import numpy as np
from .base import GreenFunction


class G_TBH_1D(GreenFunction):
    def __init__(self, E, B, R):
        if type(E) is not np.float:
            raise ValueError("Energy is not a float!")
        if type(B) is not np.float:
            raise ValueError("Bandwidth is not a float!")
        if type(R) is not np.int:
            raise ValueError("Range is not an integer!")

        self.M = np.array([])
        self.E = E
        self.B = B
        self.R = R

    def __call__(self, w=0):
        w = w + 0.000000001j
        x = (w - self.E) / (self.B)

        beta = -1j / np.sqrt(self.B ** 2 - ((w - self.E) ** 2))
        gamma = -x + 1j * np.sqrt(1 - x ** 2)

        Melements = []
        for s in range(2 * self.R + 1):
            Melements.append(beta * gamma ** s)

        G = np.zeros((2 * self.R + 1, 2 * self.R + 1), dtype=np.complex_)

        for n in range(2 * self.R + 1):
            for m in range(2 * self.R + 1):
                G[n, m] = Melements[np.abs(n - m)]
                G[m, n] = G[n, m]

        return G
