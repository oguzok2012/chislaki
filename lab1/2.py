
def ul_decompose_inplace(A):
    n = len(A)

    for i in range(n):
        # L (j >= i)
        for j in range(i, n):
            s = sum(A[i][k] * A[k][j] for k in range(i))  # U[i][k] * L[k][j]
            A[i][j] = A[i][j] - s

        # U (j > i)
        for j in range(i+1, n):
            s = sum(A[j][k] * A[k][i] for k in range(i))  # U[j][k] * L[k][i]
            A[j][i] = (A[j][i] - s) / A[i][i]

    return A

def forward_substitution(M, b):
    n = len(M)
    y = [0.0]*n
    for i in range(n):
        s = sum(M[i][k]*y[k] for k in range(i))  # U[i][k]
        y[i] = b[i] - s
    return y

def backward_substitution(M, y):
    n = len(M)
    x = [0.0]*n
    for i in reversed(range(n)):
        s = sum(M[i][j]*x[j] for j in range(i+1, n))
        x[i] = (y[i] - s) / M[i][i]
    return x

def solve_ul(A, b):
    M = ul_decompose_inplace(A)
    y = forward_substitution(M, b)
    x = backward_substitution(M, y)
    return x

def determinant_ul(M):
    det = 1.0
    for i in range(len(M)):
        det *= M[i][i]
    return det

def inverse_ul(A):
    n = len(A)
    M = ul_decompose_inplace(A)
    Inv = []
    for col in range(n):
        e = [0.0]*n
        e[col] = 1.0
        y = forward_substitution(M, e)
        x = backward_substitution(M, y)
        Inv.append(x)
    # транспонируем
    return [[Inv[j][i] for j in range(n)] for i in range(n)]

A = [
    [4, -8, -3, 7, 5],
    [8, 3, -6, -5, 2],
    [-3, 2, 4, -7, 6],
    [2, -5, -7, 6, 3],
    [7, 3, -5, 4, -9]
]
b = [-7, 17, 30, -16, -7]

M = ul_decompose_inplace([row[:] for row in A])
x = solve_ul(A, b)
detA = determinant_ul(M)
invA = inverse_ul(A)

print("M (UL-разложение):")
for row in M: print(row)
print("Решение системы:", x)
print("Определитель:", detA)
print("Обратная матрица:")
for row in invA: print(row)
