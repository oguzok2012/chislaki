import numpy as np


def solve_gauss(A, b):
    n = len(A)
    M = [row[:] for row in A]
    B = b[:]

    for k in range(n):
        max_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[max_row][k]) < 1e-12:
            raise ValueError("Система вырождена или не имеет единственного решения")
        # перестановка строк
        if max_row != k:
            M[k], M[max_row] = M[max_row], M[k]
            B[k], B[max_row] = B[max_row], B[k]

        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n):
                M[i][j] -= factor * M[k][j]
            B[i] -= factor * B[k]

    # обратный ход
    x = [0] * n
    for i in range(n - 1, -1, -1):
        s = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (B[i] - s) / M[i][i]
    return x


def determinant(A):
    n = len(A)
    M = [row[:] for row in A]
    det = 1
    sign = 1

    for k in range(n):
        max_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[max_row][k]) < 1e-12:
            return 0
        if max_row != k:
            M[k], M[max_row] = M[max_row], M[k]
            sign *= -1
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n):
                M[i][j] -= factor * M[k][j]

    for i in range(n):
        det *= M[i][i]
    return det * sign


def inverse_gauss(A):
    n = len(A)
    M = [row[:] for row in A]
    Inv = [[float(i == j) for j in range(n)] for i in range(n)]

    for k in range(n):
        max_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[max_row][k]) < 1e-12:
            raise ValueError("Матрица вырождена, обратной не существует")
        if max_row != k:
            M[k], M[max_row] = M[max_row], M[k]
            Inv[k], Inv[max_row] = Inv[max_row], Inv[k]

        pivot = M[k][k]
        for j in range(n):
            M[k][j] /= pivot
            Inv[k][j] /= pivot

        for i in range(n):
            if i != k:
                factor = M[i][k]
                for j in range(n):
                    M[i][j] -= factor * M[k][j]
                    Inv[i][j] -= factor * Inv[k][j]

    return Inv


A = [
    [4, -8, -3, 7, 5],
    [8, 3, -6, -5, 2],
    [-3, 2, 4, -7, 6],
    [2, -5, -7, 6, 3],
    [7, 3, -5, 4, -9]
]
b = [-7, 17, 30, -16, -7]

print("Решение системы:", solve_gauss(A, b))
print("Определитель:", determinant(A))
print("Обратная матрица:")
for row in inverse_gauss(A):
    print(row)
