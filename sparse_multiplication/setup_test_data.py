#!/usr/bin/env python3
import os
import subprocess
import numpy as np
import scipy.io

def id_vector_multiply(fname):
    outname = fname.split(".")[0] + "_id.mtx"
    x = scipy.io.mmread(fname)
    x = x.tocsr()
    rows, cols = x.shape
    ones = np.ones((rows, 1))

    y = x * ones

    scipy.io.mmwrite(outname, y)

if __name__ == "__main__":
    subprocess.call("./get_test_matrices.sh")
    all_files = os.listdir("data/")
    all_files = ["data/"+i for i in all_files if "_id" not in i]

    for f in all_files:
        id_vector_multiply(f)
