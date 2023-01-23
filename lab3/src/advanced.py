import numpy as np
import matplotlib.pyplot as plt


A = np.array([[3, 1, -3],
               [6, 2, 5],
               [1, 4, -3]], dtype=np.float64)

b = np.array([[-16],
               [12],
               [-39]], dtype=np.float64)


# функция LU-разложения
def lu(A, permute):
    n = len(A)
    C = np.array(A.copy())
    P = np.zeros((n, n))
    U = np.zeros((n, n), dtype=np.float64)
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            P[i][i] = 1
    for i in range(n):
        #поиск опорного элемента
        sup_el_value = 0
        sup_el = -1
        for row in range(i, n):
            if np.abs(C[row][i]) > sup_el_value:
                sup_el_value = np.abs(C[row][i])
                sup_el = row
        if sup_el_value != 0:
            if permute:
            #меняем местами i-ю строку и строку с опорным элементом
               P[[sup_el, i]] = P[[i, sup_el]]
               C[[sup_el, i]] = C[[i, sup_el]]
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
    return L, U, P


# функция решения СЛАУ
def solve(L, U, P, b):
    n = len(b)
    X = np.zeros(n)
    Y = np.zeros(n)
    b = P.dot(b)
    for i in range(n):
        Y[i] = b[i] - sum([L[i][k] * Y[k] for k in range(i)])
    for i in range(n):
        X[n - i - 1] = (Y[n - i - 1] - sum([U[n - i - 1][n - k - 1] * X[n - k - 1] for k in range(i)])) / U[n - i - 1][n - i - 1]
    return X


def norma(X):
    k = len(X)
    summary = 0
    for i in range(k):
        summary += X[i]**2
    return summary**(1/2)


if __name__ == '__main__':
    L, U, P = lu(A, permute=True)
    X = solve(L, U, P, b)
    print('solution X^:\n', X)

    A_temp = A[0][0]
    b_temp = b[0][0]
    x_tilda_false = []
    x_tilda_true = []
    y_nodes_false = []
    y_nodes_true = []
    x_nodes = []

    p = np.linspace(0, 12, 101)
    for i in range(101):
        A[0][0] = A_temp + 10**(-p[i])
        b[0][0] = b_temp + 10**(-p[i])
        L, U, P = lu(A, False)
        x_tilda_false.append(solve(L, U, P, b))
        L, U, P = lu(A, True)
        x_tilda_true.append(solve(L, U, P, b))
        y_nodes_false.append(abs(norma(X) - norma(x_tilda_false[i])) / norma(X))
        y_nodes_true.append(abs(norma(X) - norma(x_tilda_true[i])) / norma(X))
        x_nodes.append(p[i])

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.grid()
    ax.set_xlabel('$p$')
    ax.set_ylabel('$E$')
    ax.semilogy(x_nodes, y_nodes_false, 'o', color='steelblue', markersize=6, label='$E$, без частичного выбора главного элемента')
    ax.semilogy(x_nodes, y_nodes_true, 'o', color='orchid', markersize=6, label='$E$, с частичным выбором главного элемента')
    ax.legend(prop={'size': 14})
    plt.show()
