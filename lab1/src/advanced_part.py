import numpy as np
import matplotlib.pyplot as plt


#функция вычиления базисного полинома Лагранжа
def l_i(i, x, x_nodes):
    if i >= len(x_nodes):
        exit(1)
    l_i = 1
    for j in range(len(x_nodes)):
        if j!=i:
             l_i *= (x - x_nodes[j])/(x_nodes[i] - x_nodes[j])
    return l_i


#функция вычисления полинома Лагранжа
def L(x, x_nodes, y_nodes):
    L = 0
    for i in range(len(x_nodes)):
        L += y_nodes[i] * l_i(i, x, x_nodes)
    return L


#функция вычисления чебышевских узлов
def cheb(n) -> list:
    return(np.array([np.cos((2*i - 1) / (2 * n) * np.pi) for i in range(1, n + 1)]))


#класс параметров аппроксимации Паде
class Params:
    def __init__(self, n, m, a_j, b_k):
        self.n = n
        self.m = m
        self.a_j = a_j
        self.b_k = b_k


#генерируем параметры для аппроксимации Паде
def generate_params():
    n = np.random.randint(7, 16)
    m = np.random.randint(7, 16)
    a_j = np.random.sample(m, )
    b_k = np.random.sample(n, )
    return Params(n, m, a_j, b_k)


#функция вычисления значения аппроксимации Паде
def pade(x, *params_list):
    sum_a_j = 0
    sum_b_k = 0
    for obj in params_list:
        for j in range(obj.m):
            sum_a_j += obj.a_j[j] * x ** j
        for k in range(1, obj.n):
            sum_b_k += obj.b_k[k] * x ** k
    return sum_a_j / (1 + sum_b_k)


#функция вычисления и вывода аппроксимации Паде и интеполяционных полиномов
def plotting(x, x_cheb, kter):
    fig, ax = plt.subplots(1, 1, figsize=(24, 24))
    x_def = np.linspace(-1, 1, 200)
    y_def = [pade(x, params_list[kter]) for x in x_def]
    y_eq = [pade(x, params_list[kter]) for x in x_eq_nodes]
    y_eq_interp = [L(i, x_eq_nodes, y_eq) for i in x_def]
    y_cheb = [pade(x, params_list[kter]) for x in x_ch_nodes]
    y_cheb_interp = [L(i, x_ch_nodes, y_cheb) for i in x_def]
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.plot(x_eq_nodes, y_eq, 'o', color="#BA55D3", markersize=5)  # nodes
    ax.plot(x_def, y_def, '-', color="#40E0D0", label=r"$f_{n, m}(x)$")
    ax.plot(x_def, y_eq_interp, '-', color="#BA55D3", label=r"$L(x),~eq~nodes$")
    ax.plot(x_def, y_cheb_interp, '-', color="#4169E1", label=r"$L(x),~cheb~nodes$")
    ax.plot(x_ch_nodes, y_cheb, 'o', color="#4169E1", markersize=5)
    ax.grid()
    ax.legend()


#функция вычисления расстояния и построения графиков
def distance():
    nodes = [] #массив значений N = {1, ... , 30}
    max_dist_eq = [0.] * 30  # В пространстве 𝐿∞ нормой является равномерная норма,
    max_dist_ch = [0.] * 30  # которую можно определить как ||𝑔(𝑥)||∞ = max𝑥∈[𝑎;𝑏]
    for node in range(1, 31):
        nodes.append(node)
    fig, ax = plt.subplots(1, 1, figsize=(24, 24))
    for i in range(1, 31):
        x_def = np.linspace(-1, 1, 200)
        x_ch = np.array(cheb(i))  # генерируем список чебышевских узлов
        x_eq = np.linspace(-1, 1, i)  # генерируем список равномерно расположеных узлов
        y_def = [pade(x, params_list[kter]) for x in x_def]
        y_cheb = [pade(x, params_list[kter]) for x in x_ch]
        y_eq = [pade(x, params_list[kter]) for x in x_eq]
        y_eq_interpolar = [L(it, x_eq, y_eq) for it in x_def]
        y_cheb_interpolar = [L(it, x_ch, y_cheb) for it in x_def]
        max_dist_eq[i - 1] = max([np.abs(y_def[iter] - y_eq_interpolar[iter]) for iter in range(200)])
        max_dist_ch[i - 1] = max([np.abs(y_def[iter] - y_cheb_interpolar[iter]) for iter in range(200)])
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.semilogy(nodes, max_dist_eq, '-', color="#FF00FF", label=r"$||g(x)||_{\infty}~eq~nodes$")
    ax.semilogy(nodes, max_dist_ch, '-', color="#1E90FF", label=r"$||g(x)||_{\infty}~cheb~nodes$")
    ax.grid()
    ax.legend()


if __name__ == '__main__':
    #np.random.seed(1000)
    params_list = [] #список параметров для 100 функций Паде
    N = 7 #должно быть не меньше 5
    max_dist_eq = [0.] * 30   # В пространстве 𝐿∞ нормой является равномерная норма,
    max_dist_ch = [0.] * 30                      # которую можно определить как ||𝑔(𝑥)||∞ = max𝑥∈[𝑎;𝑏]
    for i in range(100): #В цикле генерируем параметры для 100 функций
        params_list.append(generate_params())
    x_ch_nodes = np.array(cheb(N)) #генерируем список чебышевских узлов
    x_eq_nodes = np.linspace(-1, 1, N) #генерируем список равномерно расположеных узлов
    for kter in range(100): #проходимся по 100 функциям
        if kter % 25 == 0: #будем выводить каждую 25-ую функцию
            plotting(x_eq_nodes, x_ch_nodes, kter)
            distance()
    plt.show()