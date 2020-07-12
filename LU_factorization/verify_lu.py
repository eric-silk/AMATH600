#!/usr/bin/env python3
import numpy as np
import scipy.linalg

if __name__ == "__main__":
    full = np.genfromtxt("full.csv", delimiter=",")
    upper = np.genfromtxt("upper.csv", delimiter=",")

    p, l, u = scipy.linalg.lu(full)

    print("Upper matches:", np.allclose(u, upper))
    print("Upper:", upper)
    print("u:", u)
