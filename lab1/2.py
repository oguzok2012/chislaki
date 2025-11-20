def ul_decompose_inplace(A):
    n = len(A)

    for i in range(n):
        # (j >= i)
        for j in range(i, n):
            s = sum(A[i][k] * A[k][j] for k in range(i))  # U[i][k] * L[k][j]
            A[i][j] = A[i][j] - s

        # (j > i)
        for j in range(i + 1, n):
            s = sum(A[j][k] * A[k][i] for k in range(i))  # U[j][k] * L[k][i]
            A[j][i] = (A[j][i] - s) / A[i][i]

    return A


def forward_substitution(M, b):
    n = len(M)
    y = [0.0] * n
    for i in range(n):
        s = sum(M[i][k] * y[k] for k in range(i))  # U[i][k]
        y[i] = b[i] - s
    return y


def backward_substitution(M, y):
    n = len(M)
    x = [0.0] * n
    for i in reversed(range(n)):
        s = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / M[i][i]
    return x


def solve_ul(A, b):
    M = ul_decompose_inplace([row[:] for row in A])  # создаем копию
    y = forward_substitution(M, b)
    x = backward_substitution(M, y)
    return x, M


def determinant_ul(M):
    det = 1.0
    for i in range(len(M)):
        det *= M[i][i]
    return det


def inverse_ul(A):
    n = len(A)
    # 1. Выполняем разложение
    M = ul_decompose_inplace([row[:] for row in A])

    inv = [[0.0] * n for _ in range(n)]

    # 2. Для каждого столбца единичной матрицы
    for j in range(n):
        # Создаем j-й столбец единичной матрицы
        e_j = [0.0] * n
        e_j[j] = 1.0

        # 3. Решаем систему A × x = e_j
        y = forward_substitution(M, e_j)
        x = backward_substitution(M, y)

        # 4. Полученное решение - j-й столбец обратной матрицы
        for i in range(n):
            inv[i][j] = x[i]

    return inv


def check_solution(A, b, x, eps=1e-10):
    """Проверка решения системы Ax = b"""
    n = len(b)
    residuals = [0.0] * n
    max_residual = 0.0

    print("\n" + "=" * 60)
    print("ПРОВЕРКА РЕШЕНИЯ СИСТЕМЫ Ax = b")
    print("=" * 60)
    print("Уравнение | Вычислено  | Ожидалось  | Ошибка")
    print("-" * 50)

    for i in range(n):
        # Вычисляем i-ю строку: A[i] * x
        computed = 0.0
        for j in range(n):
            computed += A[i][j] * x[j]

        residual = abs(computed - b[i])
        residuals[i] = residual
        max_residual = max(max_residual, residual)

        print(f"{i + 1:2d}        | {computed:10.6f} | {b[i]:10.6f} | {residual:10.2e}")

    print("-" * 50)
    print(f"Максимальное расхождение: {max_residual:.2e}")

    if max_residual < eps:
        print("✓ Решение системы корректно!")
    else:
        print("⚠ Внимание: большая невязка в решении системы!")

    return max_residual


def check_inverse(A, invA, eps=1e-10):
    """Проверка обратной матрицы A * A⁻¹ = I"""
    n = len(A)
    max_error = 0.0

    print("\n" + "=" * 60)
    print("ПРОВЕРКА ОБРАТНОЙ МАТРИЦЫ A * A⁻¹ = I")
    print("=" * 60)

    # Вычисляем A * A⁻¹
    product = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                product[i][j] += A[i][k] * invA[k][j]

    # Сравниваем с единичной матрицей
    print("Матрица A * A⁻¹ (должна быть близка к единичной):")
    for i in range(n):
        row_str = "["
        for j in range(n):
            expected = 1.0 if i == j else 0.0
            error = abs(product[i][j] - expected)
            max_error = max(max_error, error)
            row_str += f"{product[i][j]:8.4f}"
            if j < n - 1:
                row_str += " "
        row_str += "]"
        print(row_str)

    print(f"\nМаксимальное расхождние от единичной матрицы: {max_error:.2e}")

    return max_error


def check_determinant(A, det_computed, eps=1e-10):
    """Проверка определителя через свойства обратной матрицы"""
    n = len(A)

    print("\n" + "=" * 60)
    print("ПРОВЕРКА ОПРЕДЕЛИТЕЛЯ")
    print("=" * 60)

    # Проверка через обратную матрицу: det(A⁻¹) = 1/det(A)
    invA = inverse_ul([row[:] for row in A])
    det_inv = determinant_ul(ul_decompose_inplace([row[:] for row in invA]))

    if abs(det_computed) > eps:
        error = abs(1.0 / det_computed - det_inv)
        print(f"det(A) = {det_computed:.10f}")
        print(f"det(A⁻¹) = {det_inv:.10f}")
        print(f"1/det(A) = {1.0 / det_computed:.10f}")
        print(f"Разность: {error:.2e}")

    else:
        print(f"det(A) = {det_computed:.10f} (матрица вырожденная)")

    return error if abs(det_computed) > eps else 0.0



A = [
    [4, -8, -3, 7, 5],
    [8, 3, -6, -5, 2],
    [-3, 2, 4, -7, 6],
    [2, -5, -7, 6, 3],
    [7, 3, -5, 4, -9]
]
b = [-7, 17, 30, -16, -7]

print("Исходная матрица A:")
for row in A:
    print([f"{x:6.1f}" for x in row])
print(f"\nВектор b: {b}")

# Решение системы
x, M = solve_ul([row[:] for row in A], b)
detA = determinant_ul(M)
invA = inverse_ul([row[:] for row in A])

print("\nM (UL-разложение):")
for row in M:
    print([f"{x:10.6f}" for x in row])

print(f"\nРешение системы: {[f'{val:.6f}' for val in x]}")
print(f"Определитель: {detA:.6f}")

# Проверки
check_solution(A, b, x)
check_inverse(A, invA)
check_determinant(A, detA)
