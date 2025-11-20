from typing import List, Tuple


def make_diagonally_dominant(A: List[List[float]], b: List[float]) -> Tuple[List[List[float]], List[float]]:
    n = len(A)

    A_new = [row[:] for row in A]
    b_new = b[:]

    # для каждой i на диагонали ищем строку с max A[i][i]
    for i in range(n):
        max_val = abs(A_new[i][i])
        max_row = i

        # ищем строку с максимальным элементом на i-й позиции
        for j in range(i + 1, n):
            if abs(A_new[j][i]) > max_val:
                max_val = abs(A_new[j][i])
                max_row = j

        # нашли строку с большим элементом => меняем местами
        if max_row != i:
            A_new[i], A_new[max_row] = A_new[max_row], A_new[i]
            b_new[i], b_new[max_row] = b_new[max_row], b_new[i]

    return A_new, b_new


def is_diagonally_dominant(A: List[List[float]]) -> bool:
    n = len(A)

    # по строкам
    row_dominant = True
    for i in range(n):
        diagonal = abs(A[i][i])
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diagonal <= row_sum:
            row_dominant = False
            break

    # по строкам
    col_dominant = True
    for j in range(n):
        diagonal = abs(A[j][j])
        col_sum = sum(abs(A[i][j]) for i in range(n) if i != j)
        if diagonal <= col_sum:
            col_dominant = False
            break

    return row_dominant or col_dominant


def jacobi_method(
        A: List[List[float]],
        b: List[float],
        eps: float = 1e-6,
        max_iter: int = 10000
) -> Tuple[List[float], int]:
    n = len(A)

    # пытаемся сделать матрицу диагонально доминирующей
    if not is_diagonally_dominant(A):
        print("Исходная матрица не диагонально доминирующая")
        print("Пытаемся переставить строки...")
        A, b = make_diagonally_dominant(A, b)

        if not is_diagonally_dominant(A):
            print("Предупреждение: Не удалось достичь диагонального доминирования")
            print("Метод может не сходиться")

    x_old = [0.0] * n
    x_new = [0.0] * n
    iter_count = 0

    # a11*x1 + a12*x2 + .. + a1n*xn = b1
    # a21*x1 + a22*x2 + ... + a2n*xn = b2
    # ...
    # an1*x1 + an2*x2 + ... + ann*xn = bn

    #   выражаем каждую переменную через остальные
    # x1 = (b1 - a12*x2 - a13*x3 - .. - a1n*xn) / a11
    # x2 = (b2 - a21*x1 - a23*x3 - .. - a2n*xn) / a22
    # ...
    # xn = (bn - an1*x1 - an2*x2 - ... - an(n-1)*x(n-1)) / ann

    while True:
        iter_count += 1
        for i in range(n):
            s = sum(A[i][j] * x_old[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        err = max(abs(x_new[i] - x_old[i]) for i in range(n))
        if err < eps or iter_count >= max_iter:
            break
        x_old = x_new.copy()

    return x_new, iter_count
1

def print_matrix_dominance(A: List[List[float]]):
    n = len(A)
    print("Матрица A:")
    for i in range(n):
        diagonal = abs(A[i][i])
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        status = "✓" if diagonal >= row_sum else "✗"
        row_str = "  ["
        for j in range(n):
            if j == i:
                row_str += f"*{A[i][j]:6.1f}*"
            else:
                row_str += f" {A[i][j]:6.1f} "
        row_str += f"]  |{A[i][i]:.1f}| = {diagonal:.1f} >= {row_sum:.1f} {status}"
        print(row_str)


A = [
    [-7, -4, -6, 17],
    [18, 5, -7, 3],
    [-5, 6, 15, -4],
    [5, 16, -2, 7],
]

b = [89, 74, 20, -73]

print("Исходная система:")
print_matrix_dominance(A)
print(f"b = {b}")
print()

x, iterations = jacobi_method(A, b)

print("\nРезультат:")
print("Решение системы: ", x)
print("Кол-во итераций: ", iterations)


print("\nПроверка (A*x - b):")
n = len(A)
for i in range(n):
    calc = sum(A[i][j] * x[j] for j in range(n))
    error = calc - b[i]
    print(f"Уравнение {i + 1}: {calc:.2f} - {b[i]} = {error:.6f}")