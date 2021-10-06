import numpy as np
from copy import deepcopy


class GreenFunction:
    def __init__(self):
        self.M = np.array([], dtype=np.complex_)

    def __call__(self, w=0):
        return self.M

    def __add__(self, other):
        B = Gadd()
        B.A = self
        B.B = other
        return B

    def __sub__(self, other):
        B = Gsub()
        B.A = self
        B.B = other
        return B

    def __mul__(self, other):
        B = Gmul()
        B.A = self
        B.B = other
        return B

    def __truediv__(self, other):
        B = Gtruediv()
        B.A = self
        B.B = other
        return B

    def __pow__(self, other):
        pass

    def inv(self):
        return np.linalg.inv(self.M)

    def dos(self, w):
        dos = np.zeros_like(w, dtype=np.float_)

        for i in range(len(w)):
            dos[i] = -np.imag(self(w[i])) / np.pi
        return dos


class Gadd(GreenFunction):
    def __call__(self, w=0):
        return self.A(w) + self.B(w)


class Gsub(GreenFunction):
    def __call__(self, w=0):
        return self.A(w) - self.B(w)


class Gmul(GreenFunction):
    def __call__(self, w=0):
        return self.A(w).dot(self.B(w))


class Gtruediv(GreenFunction):
    def __call__(self, w=0):
        return self.A(w).dot(np.linalg.inv(self.B(w)))


class Identity(GreenFunction):
    def __init__(self, R):
        self.M = np.eye(R)


if __name__ == "__main__":
    A = np.zeros((4, 4))
    print(A())
