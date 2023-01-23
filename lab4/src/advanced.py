import numpy as np
import matplotlib.pyplot as plt
import os
from lab3_advanced import solve, lu


alpha = 3
beta = 0.002
gamma = 0.5
delta = 0.0006


# 1st ode system
def f(x):
    return np.array([alpha * x[0] - beta * x[0] * x[1], delta * x[0] * x[1] - gamma * x[1]])


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
        # print('x = ', x[:, i + 1])
    return x


def J(x):
    return np.array([[alpha - beta * x[1], -beta * x[0]], [delta * x[1], delta * x[0] - gamma]])


def newton(x_0, f, J):
    eps = 10 ** (-8)
    x_1 = x_0
    L, U, P = lu(J(x_1), True)
    s = solve(L, U, P, f(x_1))
    x_k = x_1 - s
    i = 1
    while np.linalg.norm(x_k - x_1, ord=np.inf) > eps:
        i += 1
        x_1 = x_k
        L, U, P = lu(J(x_1), True)
        s = solve(L, U, P, f(x_1))
        x_k = x_1 - s
    return x_k, i


def newton_5f(x_0, f, J):
    eps = 10 ** (-8)
    x = []
    x.append(x_0)
    L, U, P = lu(J(x_0), True)
    Y = solve(L, U, P, f(x_0))
    x_next = x_0 - Y
    x.append(x_next)
    i = 1
    k = []

    while (abs(np.linalg.norm(x[i] - x[i - 1], ord=np.inf)) > eps):
        L, U, P = lu(J(x[i]), True)
        Y = solve(L, U, P, f(x[i]))

        k_x_1 = x[i - 1]
        k_x_k = x[i]
        k.append(
            np.abs(np.linalg.norm(k_x_k, ord=np.inf) - np.linalg.norm(np.array([833.333, 1500]), ord=np.inf)) / np.abs(
                np.linalg.norm(k_x_1, ord=np.inf) - np.abs(np.linalg.norm(np.array([833.333, 1500]), ord=np.inf))) ** 2)
        x_next = x[i] - Y
        x.append(x_next)
        i += 1
    return np.array(k)


func = lambda t, x: f(x)


def g(x):
    return f(x).dot(f(x))


def count_t(x_k, z_k):
    h = lambda t: g(x_k - t * z_k / (np.linalg.norm(z_k, ord=2)))

    #метод дробления шага
    t_1 = 0.
    t_3 = 1.
    iter = 0
    while h(t_1) <= h(t_3):
        iter += 1
        t_3_d = t_3 / 2.
        t_3_u = t_3 * 2.
        if (np.linalg.norm([h(t_1) - h(t_3_d)], ord=np.inf) > np.linalg.norm([h(t_1) - h(t_3_u)], ord=np.inf)):
            t_3 = t_3_u
        else:
            t_3 = t_3_d
        if (iter > 1000):
            break
    t_2 = t_3 / 2.

    #метод наискорейшего спуска
    # t_1 = 0.
    # t_3 = 1.
    # iter = 0
    # t = np.linspace(-30, 30, num=60)
    # h_k = np.array([h(2 ** i) for i in t])
    # t_k = 2 ** t[np.argmin(h_k)]

    a, b, c = koef_search(t_1, t_2, t_3, x_k, z_k)
    t_k = (a * (t_2 + t_3) + b * (t_1 + t_3) + c * (t_1 + t_2)) / (2. * (a + b + c))
    return t_k


def koef_search(t_1, t_2, t_3, x_k, z_k):
    h = lambda t: g(x_k - t * z_k / (np.linalg.norm(z_k, ord=2)))
    a = h(t_1) / ((t_1 - t_2) * (t_1 - t_3))
    b = h(t_2) / ((t_2 - t_1) * (t_2 - t_3))
    c = h(t_3) / ((t_3 - t_1) * (t_3 - t_2))
    return a, b, c


def count_res(x_1, t_k, z_k):
    return x_1 - t_k * z_k / (np.linalg.norm(z_k, ord=2))


def gradient_descent(x_0, f, J):
    eps = 10 ** (-8)
    x_1 = x_0
    J_t = np.transpose(J(x_1))
    z_k = J_t.dot(f(x_1))
    t_k = count_t(x_1, z_k)
    x_k = count_res(x_1, t_k, z_k)
    iter = 1
    while np.linalg.norm(x_k - x_1, ord=np.inf) > eps:
        iter += 1
        x_1 = x_k
        J_t = np.transpose(J(x_1))
        z_k = J_t.dot(f(x_1))
        t_k = count_t(x_1, z_k)
        x_k = count_res(x_1, t_k, z_k)
    return x_k, iter


def gradient_descent_5f(x_0, f, J):
    eps = 10 ** (-8)
    x_1 = x_0
    J_t = np.transpose(J(x_1))
    z_k = J_t.dot(f(x_1))
    t = count_t(x_1, z_k)
    x_k = count_res(x_1, t, z_k)
    count = 1
    k = []
    while np.linalg.norm(x_k-x_1, ord=np.inf) > eps:
        count += 1

        k_x_1 = x_1
        k_x_k = x_k
        k.append(
            np.abs(np.linalg.norm(k_x_k, ord=np.inf) - np.linalg.norm(np.array([833.333, 1500]), ord=np.inf)) / np.abs(
                np.linalg.norm(k_x_1, ord=np.inf) - np.abs(np.linalg.norm(np.array([833.333, 1500]), ord=np.inf))))

        x_1 = x_k
        J_t = np.transpose(J(x_1))
        z_k = J_t.dot(f(x_1))
        t_res = count_t(x_1, z_k)
        x_k = count_res(x_1, t_res, z_k)
    #k.append(np.abs(np.linalg.norm(k_x_k, ord=np.inf)-np.linalg.norm(np.array([833.333, 1500]), ord=np.inf))/np.abs(np.linalg.norm(k_x_1, ord=np.inf)-np.abs(np.linalg.norm(np.array([833.333, 1500]), ord=np.inf))))
    return np.array(k)


def advanced_5f():
    lambd = gradient_descent_5f(np.array([2000, 300]), f, J)
    x_nodes = np.linspace(1e-6, 1e6, 1000)
    y_nodes = x_nodes**2
    x_data = np.logspace(-4, 2, len(lambd))
    y_data = lambd * x_data
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(x_nodes, x_nodes, 'green', label=r'$O(x)$')
    # ax.plot(x_nodes, y_nodes, 'red', label=r'$O(h^2)$')
    ax.plot(x_data, y_data, 'bo', label=r'$y~=~\lambda x$')
    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.loglog()
    plt.savefig('gradient_loglog.png', dpi=300)
    plt.savefig('gradient_loglog.pgf')

    lambd = newton_5f(np.array([2000, 300]), f, J)
    x_nodes = np.linspace(1e-6, 1e6, 1000)
    y_nodes = x_nodes**2
    x_data = np.logspace(-4, 2, len(lambd))
    y_data = lambd * (x_data ** 2)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    # ax.plot(x_nodes, x_nodes, 'blue', label=r'$O(h)$')
    ax.plot(x_nodes, y_nodes, 'green', label=r'$O(x^2)$')
    ax.plot(x_data, y_data, 'ro', label=r'$y~=~\lambda x^2$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    ax.legend()
    plt.loglog()
    plt.savefig('newton_loglog.png', dpi=300)
    plt.savefig('newton_loglog.pgf')

    plt.show()


def advanced():
    start_nodes = np.array([15 * i for i in range(0, 201)])
    n = len(start_nodes)
    x_nodes = np.linspace(0, 3000, n)
    y_nodes = np.linspace(0, 3000, n)
    norms_n = np.zeros((n, n))
    norms_g = np.zeros((n, n))
    count = 0
    iters_n = []
    iters_g = []
    i = 0

    for x in start_nodes:
        j = 0
        for y in start_nodes:
            count += 1
            os.system('CLS')
            print(f"Осталось: {100 - count / (n ** 2) * 100 :.{1}f}%")
            x_0 = np.array([x, y])
            # print(x_0)
            root_n, iter_n = newton(x_0, f, J)
            iters_n.append(iter_n)
            # print('newton: root_n =', root_n, 'iter_n = ', iter_n)
            root_g, iter_g = gradient_descent(x_0, f, J)
            iters_g.append(iter_g)
            # print('gradient: root_g =', root_g, 'iter_g = ', iter_g)
            #print(root_g)
            # norm_n = np.linalg.norm(root_n, ord=np.inf)  #хз как
            norm_g = np.linalg.norm(root_g, ord=np.inf)
            # norms_n[i, j] = norm_n
            norms_g[i, j] = norm_g
            # print('gradient norm = ', norms_g[i, j])
            j+=1
        i+=1
        # print('i = ', i)
    #print(norms)

    M_n = sum(iters_n) / len(iters_n)
    M_g = sum(iters_g) / len(iters_g)

    S_n = np.sqrt((1 / (len(iters_n) * (len(iters_n) - 1))) * sum([(iter - M_n) ** 2 for iter in iters_n]))
    S_g = np.sqrt((1 / (len(iters_g) * (len(iters_g) - 1))) * sum([(iter - M_g) ** 2 for iter in iters_g]))

    print('M_n = ', M_n, 'S_n = ', S_n)
    print('M_g = ', M_g, 'S_g = ', S_g)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.contourf(norms_n)
    cs = ax.contourf(x_nodes, y_nodes, norms_n, cmap="PuBu_r")
    cbar = plt.colorbar(cs)
    #ax.contourf([nodes, nodes,], norms)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    cs = ax.contourf(x_nodes, y_nodes, norms_g, cmap="PuBu_r")
    cbar = plt.colorbar(cs)
    # ax.contourf(norms_g)
    plt.show()
    print('end')
    #print(iter, )


if __name__ == '__main__':
    # advanced()
    advanced_5f()