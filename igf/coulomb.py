#
from .base import GreenFunction
import numpy as np


class CoulombHamiltonian(GreenFunction):
    def __init__(self, delta, A, R):

        if type(delta) is not np.float:
            raise ValueError("Energy is not a float!")
        if type(A) is not np.float:
            raise ValueError("Bandwidth is not a float!")
        if type(R) is not np.int:
            raise ValueError("Range is not an integer!")

        Hdiag = np.zeros((2 * R + 1,))
        for s in range(2 * R + 1):
            r = s - R
            if r == 0:
                Hdiag[s] = -A - delta
            else:
                Hdiag[s] = -A / np.abs(r)

        self.M = np.diag(Hdiag)
