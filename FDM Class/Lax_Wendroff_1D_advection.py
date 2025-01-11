import numpy as np
import matplotlib.pyplot as plt

a = 1.0
T = 2.0
N = [49, 99, 199, 399, 799]


def U_0(x):
    return 2 * np.exp(-200 * (x - 0.5)**2) * np.cos(40 * np.pi * x)


def Lax_Wendroff(a, N, T):
    h = 1.0 / (N + 1)
    k = h / 2
    x = np.linspace(0, 1, N + 2)
    U_0_arr = U_0(x)
    U_n = U_0_arr
    U_n_prev = U_0_arr
    U_n[0] = U_n_prev[N]
    U_n_prev[N + 1] = U_n_prev[1]

    for t in np.arange(1, T):
        U_n[1:N + 1] = U_n_prev[1:N + 1] - (a * k / (6 * h)) * (3 * U_n_prev[1:N + 1] - 6 * U_n_prev[0:N] + U_n_prev[N - 1:-2] + 2 * U_n_prev[2:N + 2]) + (a**2 * k**2 / (2 * h**2)) * (U_n_prev[0:N] + 2 * U_n_prev[1:N + 1] + U_n_prev[2:N + 2]) + (a**3 * k**3 / (6 * h**3)) * (3 * U_n_prev[0:N] - U_n_prev[N - 1:-2] - 3 * U_n_prev[1:N + 1] + U_n_prev[2:N + 2])
        U_n[0] = U_n_prev[N]
        U_n_prev[N + 1] = U_n_prev[1]
        U_n_prev = U_n

    errors = np.linalg.norm(U_n - U_0(x)) / np.linalg.norm(U_0(x))
    print(errors)

    plt.plot(x, U_n, label='Numerical N=%s' % N)
    plt.plot(x, U_0(x), '--', label='Exact')
    plt.title('Advection equation solved using Lax-Wendroff method for N=%d' % N)
    plt.xlabel('x')
    plt.ylabel('U(x,t)')
    plt.legend()
    plt.show()


for n in N:
    Lax_Wendroff(a, n, T)
