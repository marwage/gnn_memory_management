import numpy as np


def permutate(old_idx, N, M):
    last_idx = M * N - 1
    if old_idx == last_idx:
        return last_idx
    else:
        return (N * old_idx) % last_idx


N, M = 5, 4
a = np.random.rand(N, M)
print("a")
print(a)

a_T = a.T
a_T_flat = a_T.flatten()
a_flat = a.flatten()


a_T_self_flat = np.zeros((M * N,))
for i in range(N):
    for j in range(M):
        old_idx = M * i + j
        new_idx = permutate(old_idx, N, M)
        a_T_self_flat[new_idx] = a_flat[old_idx];

print("a_T_flat")
print(a_T_flat)
print("a_T_self_flat")
print(a_T_self_flat)

a_T_self = a_T_self_flat.reshape((M, N))
print("a_T_self")
print(a_T_self)
print("a_T")
print(a_T)

new_idx = permutate(24062, N, M)
print("new index {}".format(new_idx))

N, M = 2, 5
b = np.array([0,1,2,3,4,5,6,7,8,9])
b_mat = b.reshape((N, M))
print("b")
print(b_mat)

for i in range(N):
    for j in range(M):
        old_idx = M * i + j
        new_idx = permutate(old_idx, N, M)
        b[new_idx] = b[old_idx];

b = b.reshape((M, N))

print("b_T_self")
print(b)

