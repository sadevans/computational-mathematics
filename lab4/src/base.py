import numpy as np
import matplotlib.pyplot as plt

a = 3
b = 0.002
g = 0.5
d = 0.0006


#1st ode system
def func(x):
    return np.array([a * x[0] - b * x[0] * x[1], d * x[0] * x[1] - g * x[1]])


def rk4(x_0, t_n, f, h):
    t = np.arange(0, t_n, h)
    n = t.size
    kolvo = x_0.size
    x = np.zeros((kolvo, n))
    x[:, 0] = x_0
    for i in range(n - 1):
        k1 = h * f(t[i], x[:, i])
        k2 = h * f(t[i] + h / 2, x[:, i] + k1 / 2)
        k3 = h * f(t[i] + h / 2, x[:, i] + k2 / 2)
        k4 = h * f(t[i] + h, x[:, i] + k3)
        koef = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x[:, i + 1] = x[:, i] + koef
    return x


if __name__ == '__main__':
    f = lambda t, x: func(x)
    t_n = 20
    h = 0.01
    time = np.arange(0, t_n, h)
    start_nodes = np.array([i * 200 for i in range(1, 11)])
    # x_0 = [200, 200]
    print(start_nodes)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.grid()
    for x_1 in start_nodes:
        for y_1 in start_nodes:
            x_0 = np.array([x_1, y_1])
            res = rk4(x_0, t_n, f, h)
            ax.set_xlabel('Количество жертв $x$')
            ax.set_ylabel('Количество хищников $y$')
            ax.plot(res[0, :], res[1, :], '-', color='steelblue')
    # plt.savefig('phas_portrait.png', dpi=300)
    ax.plot(g/d, a/b, 'ro')
    # ax.plot(0, 0, 'ro')
    plt.savefig('portrait_dot.png', dpi=300)
            # ax.legend()
    # # x_0 = [200, 200]
    # print('begining here')
    # x_0 = np.array([200, 200])
    # print('res = ', rk4(x_0, t_n, f, h))
            # ax.plot(t, res[0, :], color='blue')
    # plt.show()
    x_0 = np.array([1000, 500])
    res1 = rk4(x_0, t_n, f, h)
    print('res[0, :]', res1[0, :])
    print('res[1, :]', res1[1, :])
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.grid()
    ax.set_xlabel('t')
    ax.set_ylabel(r'$x,~y$')
    ax.plot(time, res[0, :], color='cornflowerblue', label=r'$x(t)$')
    ax.plot(time, res[1, :], color='indigo', label=r'$y(t)$')
    ax.legend()
    plt.savefig('xy_graf.png', dpi=300)

    plt.show()