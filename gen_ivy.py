import numpy as np
import scipy.sparse as sp
import scipy.io


def gen_data():
    dir_path = "/mnt/data/ivy"

    n = 2 ** 23
    print("n {}".format(n))
    f = 512
    print("f {}".format(f))
    c = 64
    print("c {}".format(c))
    density = 1e-5

    classes = np.random.randint(0, c, n)
    classes = classes.astype(np.int32)
    path = dir_path + "/classes.npy"
    np.save(path, classes)

    return # delete again

    features = np.random.randn(n, f)
    features = features.astype(np.float32)
    path = dir_path + "/features.npy"
    np.save(path, features)

    k = int(n * n * density)
    print("k {}".format(k))
    values = np.ones(k, dtype=np.float32)
    i = np.random.randint(n, size=k, dtype=np.int32)
    j = np.random.randint(n, size=k, dtype=np.int32)
    adj = sp.coo_matrix((values, (i, j)), shape=(n, n)).asformat("csr")

    path = dir_path + "/adjacency.mtx"
    scipy.io.mmwrite(path, adj)


if __name__ == "__main__":
    gen_data()

