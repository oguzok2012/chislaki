from typing import List, Tuple


def check_thomas_stability(A: List[List[float]]) -> bool:
    n = len(A)

    a = [0.0] + [A[i][i - 1] for i in range(1, n)]
    b = [A[i][i] for i in range(n)]
    c = [A[i][i + 1] for i in range(n - 1)] + [0.0]

    # |b_i| >= |a_i| + |c_i|
    cond1 = all(abs(b[i]) >= abs(a[i]) + abs(c[i]) for i in range(n))

    # |c₁/b₁| < 1
    cond2 = abs(c[0] / b[0]) < 1 if n > 0 else True

    # |a_n/b_n| < 1
    cond3 = abs(a[n - 1] / b[n - 1]) < 1 if n > 0 else True

    stable = cond1 and cond2 and cond3
    return stable


def thomas_solve_and_det(A, d):
    if not check_thomas_stability(A):
        print("Метод может быть неустойчив")

    n = len(d)
    a = [0.0] + [A[i][i - 1] for i in range(1, n)]
    b = [A[i][i] for i in range(n)]
    c = [A[i][i + 1] for i in range(n - 1)] + [0.0]
    d = d[:]

    for i in range(1, n):
        w = a[i] / b[i - 1]  # w = aᵢ / bᵢ₋₁
        b[i] = b[i] - w * c[i - 1]  # bᵢ = bᵢ - w × cᵢ₋₁
        d[i] = d[i] - w * d[i - 1]  # dᵢ = dᵢ - w × dᵢ₋₁

    x = [0.0] * n
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    det = 1.0
    for bi in b:
        det *= bi

    return x, det


def check_solution(A, d, x, eps=1e-10):
    n = len(d)
    residuals = [0.0] * n
    max_residual = 0.0

    print("\nПроверка решения:")
    print("Уравнение | Вычислено  | Ожидалось  | Ошибка")
    print("-" * 50)

    for i in range(n):
        # Вычисляем i-ю строку: A[i] * x
        computed = 0.0
        for j in range(n):
            computed += A[i][j] * x[j]

        residual = abs(computed - d[i])
        residuals[i] = residual
        max_residual = max(max_residual, residual)

        print(f"{i + 1:2d}        | {computed:10.5f} | {d[i]:10.5f} | {residual:10.2e}")

    print("-" * 50)
    print(f"Максимальная невязка: {max_residual:.2e}")

    if max_residual < eps:
        print("Решение корректно!")
    else:
        print("Внимание: большое расхождение!")

    return max_residual


# Основная программа
A = [
    [7, -3, 0, 0, 0, 0, 0, 0],
    [3, 9, -2, 0, 0, 0, 0, 0],
    [0, 5, 10, -4, 0, 0, 0, 0],
    [0, 0, -2, 8, -3, 0, 0, 0],
    [0, 0, 0, 3, 9, 4, 0, 0],
    [0, 0, 0, 0, -5, 10, 4, 0],
    [0, 0, 0, 0, 0, 3, -7, -2],
    [0, 0, 0, 0, 0, 0, 3, 8],
]
d = [54, -24, -10, 22, 51, -40, 33, -7]

print("Решение системы методом Томаса:")
x, detA = thomas_solve_and_det(A, d)

print("Решение системы:")
for i, val in enumerate(x):
    print(f"x[{i}] = {val:.4f}")

print(f"\nОпределитель: {detA:.4f}")

# Проверка решения
check_solution(A, d, x)

