import numpy as np
import matplotlib.pyplot as plt

def l_i(i, x, x_nodes):
    l_i = 1
    for j in range(len(x_nodes)):
        if j!=i:
             l_i *= (x - x_nodes[j])/(x_nodes[i] - x_nodes[j])
    return l_i


def L(x, x_nodes, y_nodes):
    L = 0
    for i in range(len(x_nodes)):
        L += y_nodes[i] * l_i(i, x, x_nodes)
    return L


def plot(x_eq_nodes, x_ch_nodes, f):
    y_eq_nodes = f(x_eq_nodes)
    y_ch_nodes = f(x_ch_nodes)
    fig, ax = plt.subplots(1, 1, figsize = (12, 12))
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    x_for_plotting = np.linspace(-1, 1, 200)
    ax.plot(x_eq_nodes, y_eq_nodes, 'go', markersize=10)
    ax.plot(x_for_plotting, f(x_for_plotting), '-', color = '#aaa', linewidth = 4, label='f(x)')
    ax.plot(x_for_plotting, [L(x, x_eq_nodes, y_eq_nodes) for x in x_for_plotting], 'g-', linewidth = 2, label='L(x), eq nodes')
    ax.plot(x_ch_nodes, y_ch_nodes, 'bo', markersize=10)
    ax.plot(x_for_plotting, [L(x, x_ch_nodes, y_ch_nodes) for x in x_for_plotting], 'b-', linewidth=2, label='L(x), cheb nodes')
    ax.grid()
    ax.legend()
    plt.show()


def cheb(n) -> list:
    return(np.array([np.cos((2*i - 1) / (2 * n) * np.pi) for i in range(1, n + 1)]))

f = lambda x: 1./(1. + 25*x**2)
n_iter = np.array([5, 8, 11, 14, 17, 20, 23])
for a in range(len(n_iter)):
    x_equi_nodes = np.linspace(-1, 1, n_iter[a])
    x_ch_nodes = cheb(n_iter[a])
    plot(x_equi_nodes, x_ch_nodes, f)
