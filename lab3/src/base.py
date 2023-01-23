import numpy as np


A = np.array([[1, 1, 0, 3],
               [2, 1, -1, 1],
               [3, -1, -1, 2],
               [-1, 2, 3, -1]], dtype = float)

b = np.array([[4],
               [1],
               [-3],
               [4]], dtype = float)


def lu(A):
    n = len(A)
    C = np.array(A.copy())
    U = np.zeros((n, n), dtype=np.float64)
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            C[j][i] /= C[i][i]
            for k in range(i + 1, n):
                C[j][k] -= C[j][i] * C[i][k]
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i][j] = C[i][j]
            if i == j:
                L[i][j] = 1
                U[i][j] = C[i][j]
            if i < j:
                U[i][j] = C[i][j]
    return L, U


def solve(L, U, b):
    n = len(b)
    X = np.zeros(n)
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = b[i] - sum([L[i][k] * Y[k] for k in range(i)])
    for i in range(n):
        summa = sum([U[n - i - 1][n - k - 1] * X[n - k - 1] for k in range(i)])
        y = Y[n - i - 1]
        X[n - i - 1] = (y - summa) / U[n - i - 1][n - i - 1]
    return X


if __name__ == '__main__':
    L, U = lu(A)
    X = solve(L, U, b)
    print(X)

