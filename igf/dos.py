import numpy as np


def DOS(G, w, pos=None):
    dos = np.zeros_like(w, dtype=np.float_)

    sG = np.shape(G())
    for i in range(len(w)):
        if pos is None:
            dos[i] = -np.imag(np.trace(G(w[i]))) / np.pi / sG[0]
        else:
            dos[i] = -np.imag(G(w[i])[pos]) / np.pi

    return dos
