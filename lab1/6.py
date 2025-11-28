import math
import cmath
from typing import List, Tuple
import numpy as np
l = 7


class Vector:
    def __init__(self, data: List[float]):
        self.data = data
        self.n = len(data)

    def to_matrix(self):
        return Matrix([[x] for x in self.data])

    def scale(self, factor: float):
        return Vector([x * factor for x in self.data])

    def add(self, other):
        if self.n != other.n:
            raise ValueError("Vectors must have same length")
        return Vector([self.data[i] + other.data[i] for i in range(self.n)])

    def print_vector(self, name: str = ""):
        if name:
            print(f"{name} = ")
        print("[" + " ".join(f"{x:10.6f}" for x in self.data) + "]")


class CmpVector:
    def __init__(self, data: List[complex]):
        self.data = data
        self.n = len(data)

    def add(self, other):
        if self.n != other.n:
            raise ValueError("Vectors must have same length")
        return CmpVector([self.data[i] + other.data[i] for i in range(self.n)])

    def scale(self, factor: complex):
        return CmpVector([x * factor for x in self.data])

    def print_vector(self, name: str = ""):
        if name:
            print(f"{name} = ")
        for i, val in enumerate(self.data):
            if abs(val.imag) < 1e-10:
                print(f"[{i}]: {val.real:.6f}")
            else:
                print(f"[{i}]: {val.real:.6f} + {val.imag:.6f}i")


class Matrix:
    def __init__(self, data: List[List[float]]):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0
        self.n = self.rows

    def is_square(self) -> bool:
        return self.rows == self.cols

    def copy(self):
        return Matrix([row[:] for row in self.data])

    def transpose(self):
        result = [[0.0] * self.rows for _ in range(self.cols)]
        for i in range(self.rows):
            for j in range(self.cols):
                result[j][i] = self.data[i][j]
        return Matrix(result)

    def multiply(self, other):
        if self.cols != other.rows:
            raise ValueError("Matrix dimensions don't match for multiplication")

        result = [[0.0] * other.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result[i][j] += self.data[i][k] * other.data[k][j]
        return Matrix(result)

    def add(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for addition")

        result = [[0.0] * self.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                result[i][j] = self.data[i][j] + other.data[i][j]
        return Matrix(result)

    def scale(self, factor: float):
        result = [[0.0] * self.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                result[i][j] = self.data[i][j] * factor
        return Matrix(result)

    def print_matrix(self, name: str = ""):
        if name:
            print(f"{name}:")
        for row in self.data:
            print("[" + " ".join(f"{x:10.6f}" for x in row) + "]")
        print()


def identity(n: int) -> Matrix:
    data = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    return Matrix(data)


def sign(x: float) -> int:
    return -1 if x < 0 else 1


def frobenius_norm(v: Vector) -> float:
    return math.sqrt(sum(x * x for x in v.data))


def householder(v: Vector) -> Matrix:
    n = v.n
    v_matrix = v.to_matrix()
    v_transpose = v_matrix.transpose()

    # v * v^T
    v_vt = v_matrix.multiply(v_transpose)

    # v^T * v
    vt_v = v_transpose.multiply(v_matrix).data[0][0]

    # H = I - 2 * (v * v^T) / (v^T * v)
    H = identity(n).add(v_vt.scale(-2.0 / vt_v))

    return H


def householder_qr(A: Matrix, eps: float = 1e-12) -> Tuple[Matrix, Matrix]:
    """QR-разложение с преобразованиями Хаусхолдера с проверкой нормы """
    n = A.n
    Q = identity(n)
    R = A.copy()

    for k in range(n - 1):
        # Вектор x из k-го столбца начиная с k-й строки
        x = Vector([R.data[i][k] for i in range(k, n)])
        norm_x = frobenius_norm(x)

        # Проверка
        if norm_x < eps:
            continue

        # Вектор Хаусхолдера
        v_data = [0.0] * n
        for i in range(k):
            v_data[i] = 0.0

        # v[0] = x[0] - sign(x[0]) * norm_x
        v_data[k] = x.data[0] + sign(x.data[0]) * norm_x

        for i in range(k + 1, n):
            v_data[i] = x.data[i - k]

        v = Vector(v_data)
        norm_v = frobenius_norm(v)

        # Проверка
        if norm_v < eps:
            continue

        # Нормализация вектора v
        v_normalized = v.scale(1.0 / norm_v)

        H = householder(v_normalized)

        # R = H * R
        R = H.multiply(R)
        # Q = Q * H
        Q = Q.multiply(H)

    return Q, R


def solve_quadratic(a: float, b: float, c: float) -> Tuple[complex, complex]:
    D = complex(b * b - 4 * a * c, 0)
    sqrt_D = cmath.sqrt(D)
    denominator = complex(2 * a, 0)

    x1 = (-complex(b, 0) + sqrt_D) / denominator
    x2 = (-complex(b, 0) - sqrt_D) / denominator

    return x1, x2


def extract_eigenvalues(A: Matrix, eps: float = 1e-12) -> CmpVector:
    """Извлечение собственных значений из почти треугольной матрицы"""
    n = A.n
    result = CmpVector([complex(0, 0)] * n)
    i = 0

    while i < n:
        if i == n - 1:
            result.data[i] = complex(A.data[i][i], 0)
            i += 1
            continue
        else:
            # блок - 2x2 не диагональный
            if abs(A.data[i][i + 1]) < eps and abs(A.data[i + 1][i]) < eps:
                # Вещественные собственные значения
                result.data[i] = complex(A.data[i][i], 0)
                i += 1
            else:
                a11, a12 = A.data[i][i], A.data[i][i + 1]
                a21, a22 = A.data[i + 1][i], A.data[i + 1][i + 1]

                # λ² - (a11 + a22)λ + (a11*a22 - a12*a21) = 0
                coefs = [1.0, -(a11 + a22), a11 * a22 - a12 * a21]
                root1, root2 = solve_quadratic(coefs[0], coefs[1], coefs[2])

                result.data[i] = root1
                result.data[i + 1] = root2
                i += 2
                continue

    return result


def cmp_infinity_norm(v: CmpVector) -> float:
    return max(abs(x) for x in v.data)


def get_eigenvalues(A: Matrix, eps: float, prev_eigenvalues: CmpVector = None) -> Tuple[CmpVector, bool]:
    n = A.n
    eigenvalues = extract_eigenvalues(A, eps)

    if prev_eigenvalues is None:
        return eigenvalues, False

    # Проверяем сходимость всех собственных значений
    end = True
    for i in range(n):
        if abs(eigenvalues.data[i] - prev_eigenvalues.data[i]) > eps:
            end = False
            break

    return eigenvalues, end


def qr_eigenvalues(A: Matrix, eps: float = 1e-6, max_iter: int = 1000) -> Tuple[CmpVector, int]:
    n = A.n
    Ak = A.copy()

    iter_count = 0
    prev_eigenvalues = None
    eigenvalues = None

    for iter_count in range(1, max_iter + 1, l):
        Q, R = householder_qr(Ak, eps)

        # A_{k+1} = R * Q
        Ak = R.multiply(Q)

        # Получаем собственные значения и проверяем сходимость
        eigenvalues, stop = get_eigenvalues(Ak, eps, prev_eigenvalues)

        if stop:
            print(f"Сходимость достигнута за {iter_count} итераций")
            break

        prev_eigenvalues = eigenvalues

    if iter_count == max_iter:
        print(f"Достигнуто максимальное количество итераций: {max_iter}")

    print("\nИтоговая матрица A после QR-алгоритма:")
    Ak.print_matrix()
    return eigenvalues, iter_count


# Основная программа
if __name__ == "__main__":
    A_data = [
        [15, -5, -4, 1, 0],
        [4, 12, 0, 2, 2],
        [-2, 3, 7, 5, 5],
        [11, 5, -4, 5, 3],
        [1, 3, -5, 1, -2]
    ]

    A = Matrix(A_data)

    print("QR-АЛГОРИТМ С ПРЕОБРАЗОВАНИЕМ ХАУСХОЛДЕРА")
    A_np = np.array(A_data)
    print("Исходная матрица A:")
    print(A_np)

    eigenvalues, iterations = qr_eigenvalues(A, eps=0.0001, max_iter=1000)

    print(f"Количество итераций: {iterations}")

    print("\n" + "=" * 60)
    print("\nСобственные значения (по алгоритму):")
    eigenvalues.print_vector()


    # Собственные значения через numpy
    # eigenvalues_np = np.linalg.eigvals(A_np)
    # print("\nСобственные значения (numpy):")
    # for i, val in enumerate(eigenvalues_np):
    #     if abs(val.imag) < 1e-10:
    #         print(f"[{i}]: {val.real:.6f}")
    #     else:
    #         print(f"[{i}]: {val.real:.6f} + {val.imag:.6f}i")