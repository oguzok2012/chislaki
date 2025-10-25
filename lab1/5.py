from typing import List, Tuple


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

    if not row_dominant:
        return False

    # по столбцам
    for j in range(n):
        diagonal = abs(A[j][j])
        col_sum = sum(abs(A[i][j]) for i in range(n) if i != j)
        if diagonal <= col_sum:
            col_dominant = False
            break

    return row_dominant or col_dominant


def make_diagonally_dominant(A: List[List[float]], b: List[float]) -> Tuple[List[List[float]], List[float]]:
    n = len(A)

    A_new = [row[:] for row in A]
    b_new = b[:]

    for i in range(n):
        max_val = abs(A_new[i][i])
        max_row = i

        for j in range(i + 1, n):
            if abs(A_new[j][i]) > max_val:
                max_val = abs(A_new[j][i])
                max_row = j

        if max_row != i:
            A_new[i], A_new[max_row] = A_new[max_row], A_new[i]
            b_new[i], b_new[max_row] = b_new[max_row], b_new[i]

    return A_new, b_new

def seidel_method(
        A: List[List[float]],
        b: List[float],
        eps: float = 0.0001,
        max_iter: int = 1000
) -> Tuple[List[float], int]:
    n = len(A)

    if not is_diagonally_dominant(A):
        print("Матрица не диагонально доминирующая")
        print("Пытаемся переставить строки")
        A, b = make_diagonally_dominant(A, b)

        if not is_diagonally_dominant(A):
            print("Предупреждение: Не удалось достичь строгого диагонального преобладания")
            print("Метод может сходиться медленно или не сходиться")

    x = [0.0] * n
    iter_count = 0

    for iteration in range(max_iter):
        iter_count += 1
        max_error = 0.0

        for i in range(n):
            sum1 = sum(A[i][j] * x[j] for j in range(i))
            sum2 = sum(A[i][j] * x[j] for j in range(i + 1, n))

            new_x = (b[i] - sum1 - sum2) / A[i][i]

            error = abs(new_x - x[i])
            if error > max_error:
                max_error = error

            x[i] = new_x

        if max_error < eps:
            break

    return x, iter_count


def print_system(A: List[List[float]], b: List[float]):
    n = len(A)
    variables = ['x₁', 'x₂', 'x₃', 'x₄']

    print("Система уравнений:")
    for i in range(n):
        equation = ""
        for j in range(n):
            if A[i][j] != 0:
                sign = "+" if A[i][j] > 0 and j > 0 else ""
                equation += f"{sign}{A[i][j]:g}{variables[j]} "
        equation += f"= {b[i]}"
        print(equation)


A = [
    [-7, -4, -6, 17],
    [18, 5, -7, 3],
    [-5, 6, 15, -4],
    [5, 16, -2, 7],
]

b = [89, 74, 20, -73]


print("Проверка диагонального преобладания:")
for i in range(len(A)):
    diagonal = abs(A[i][i])
    row_sum = sum(abs(A[i][j]) for j in range(len(A)) if j != i)
    status = "✓" if diagonal >= row_sum else "✗"
    print(f"Уравнение {i + 1}: |{A[i][i]}| = {diagonal} >= {row_sum} {status}")

print("\n" + "=" * 50)


print_system(A, b)
print(f"\nТочность ε = {0.0001}")

x, iterations = seidel_method(A, b, eps=0.0001)

print(f"\nРезультаты:")
print(f"Количество итераций: {iterations}")
print(f"Решение:")
for i, val in enumerate(x):
    print(f"x{i + 1} = {val:.6f}")

print(f"\nПроверка решения (подстановка в уравнения):")
for i in range(len(A)):
    calculated = sum(A[i][j] * x[j] for j in range(len(A)))
    error = abs(calculated - b[i])
    print(f"Уравнение {i + 1}: {calculated:.6f} ≈ {b[i]} (погрешность: {error:.6f})")