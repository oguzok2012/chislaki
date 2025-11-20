import math
import matplotlib.pyplot as plt

# Исходные данные
xs = [-4.14, -3.16, -2.18, -1.20, -0.22, 0.76, 1.74, 2.72, 3.70]
ys = [-2.3285, -0.7683, -1.2326, 0.4502, 1.0379, 2.9643, 2.8924, 3.3481, 2.7326]
xStar = 1.493

# Варианты выбора узлов
quadraticVariant = 1  # 1=LeftInterval, 2=RightInterval
cubicVariant = 2  # 1=LeftAndTwoRight, 2=Middle, 3=RightAndTwoSides


def dividedDifferences(X, Y):
    """Вычисление разделенных разностей"""
    n = len(X)
    table = [[0.0] * n for _ in range(n)]

    for i in range(n):
        table[i][0] = Y[i]

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (X[i + j] - X[i])

    return table[0]


def evaluateNewton(X, coeffs, x):
    """Вычисление значения полинома Ньютона в точке x"""
    result = coeffs[0]
    product = 1.0
    for i in range(1, len(coeffs)):
        product *= (x - X[i - 1])
        result += coeffs[i] * product
    return result


def buildPolynomial2LeftInterval(xStar):
    """Построение квадратичного полинома - левый интервал"""
    leftIdx = 0
    for i in range(len(xs) - 1):
        if xs[i] <= xStar and xs[i + 1] >= xStar:
            leftIdx = i
            break

    xNodes = [xs[leftIdx - 1], xs[leftIdx], xs[leftIdx + 1]]
    yNodes = [ys[leftIdx - 1], ys[leftIdx], ys[leftIdx + 1]]
    coeffs = dividedDifferences(xNodes, yNodes)
    value = evaluateNewton(xNodes, coeffs, xStar)
    return xNodes, yNodes, value, coeffs


def buildPolynomial2RightInterval(xStar):
    """Построение квадратичного полинома - правый интервал"""
    leftIdx = 0
    for i in range(len(xs) - 1):
        if xs[i] <= xStar and xs[i + 1] >= xStar:
            leftIdx = i
            break

    xNodes = [xs[leftIdx], xs[leftIdx + 1], xs[leftIdx + 2]]
    yNodes = [ys[leftIdx], ys[leftIdx + 1], ys[leftIdx + 2]]
    coeffs = dividedDifferences(xNodes, yNodes)
    value = evaluateNewton(xNodes, coeffs, xStar)
    return xNodes, yNodes, value, coeffs


def buildPolynomial3LeftAndTwoRight(xStar):
    """Построение кубического полинома - слева и два справа"""
    leftIdx = 0
    for i in range(len(xs) - 1):
        if xs[i] <= xStar and xs[i + 1] >= xStar:
            leftIdx = i
            break

    xNodes = [xs[leftIdx], xs[leftIdx + 1], xs[leftIdx + 2], xs[leftIdx + 3]]
    yNodes = [ys[leftIdx], ys[leftIdx + 1], ys[leftIdx + 2], ys[leftIdx + 3]]
    coeffs = dividedDifferences(xNodes, yNodes)
    value = evaluateNewton(xNodes, coeffs, xStar)
    return xNodes, yNodes, value, coeffs


def buildPolynomial3Middle(xStar):
    """Построение кубического полинома - средний вариант"""
    leftIdx = 0
    for i in range(len(xs) - 1):
        if xs[i] <= xStar and xs[i + 1] >= xStar:
            leftIdx = i
            break

    xNodes = [xs[leftIdx - 1], xs[leftIdx], xs[leftIdx + 1], xs[leftIdx + 2]]
    yNodes = [ys[leftIdx - 1], ys[leftIdx], ys[leftIdx + 1], ys[leftIdx + 2]]
    coeffs = dividedDifferences(xNodes, yNodes)
    value = evaluateNewton(xNodes, coeffs, xStar)
    return xNodes, yNodes, value, coeffs


def buildPolynomial3RightAndTwoSides(xStar):
    """Построение кубического полинома - справа и два по бокам"""
    leftIdx = 0
    for i in range(len(xs) - 1):
        if xs[i] <= xStar and xs[i + 1] >= xStar:
            leftIdx = i
            break

    xNodes = [xs[leftIdx - 2], xs[leftIdx - 1], xs[leftIdx], xs[leftIdx + 1]]
    yNodes = [ys[leftIdx - 2], ys[leftIdx - 1], ys[leftIdx], ys[leftIdx + 1]]
    coeffs = dividedDifferences(xNodes, yNodes)
    value = evaluateNewton(xNodes, coeffs, xStar)
    return xNodes, yNodes, value, coeffs


def buildPolynomial(degree, xStar):
    """Построение полинома заданной степени"""
    if degree == 2:
        if quadraticVariant == 2:
            xNodes, yNodes, value, coeffs = buildPolynomial2RightInterval(xStar)
        else:
            xNodes, yNodes, value, coeffs = buildPolynomial2LeftInterval(xStar)
    else:
        if cubicVariant == 2:
            xNodes, yNodes, value, coeffs = buildPolynomial3Middle(xStar)
        elif cubicVariant == 3:
            xNodes, yNodes, value, coeffs = buildPolynomial3RightAndTwoSides(xStar)
        else:
            xNodes, yNodes, value, coeffs = buildPolynomial3LeftAndTwoRight(xStar)

    return xNodes, yNodes, value, coeffs


def newtonPolynomialString(X, coeffs):
    """Формирование строкового представления полинома Ньютона"""
    result = f"{coeffs[0]:.6f}"
    for i in range(1, len(coeffs)):
        if coeffs[i] >= 0:
            result += " + "
        else:
            result += " - "
        result += f"{abs(coeffs[i]):.6f}"
        for j in range(i):
            if X[j] >= 0:
                result += f"(x-{X[j]:.2f})"
            else:
                result += f"(x+{abs(X[j]):.2f})"
    return result


def estimateMaxDerivative(X, Y):
    """Оценка максимальной производной через разделенные разности"""
    n = len(X)
    table = [[0.0] * n for _ in range(n)]

    for i in range(n):
        table[i][0] = Y[i]

    maxDiff = 0.0
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (X[i + j] - X[i])
            if abs(table[i][j]) > maxDiff:
                maxDiff = abs(table[i][j])

    return maxDiff * 10  # Эмпирический коэффициент


def estimateError(X, x, maxDerivative):
    """Оценка погрешности интерполяции"""
    omega = 1.0
    for i in range(len(X)):
        omega *= abs(x - X[i])

    n = len(X) - 1
    factorial = 1.0
    for i in range(2, n + 2):
        factorial *= i

    return (maxDerivative / factorial) * omega


def create_plot():
    """Создание графика"""
    plt.figure(figsize=(12, 8))

    # Получаем данные для полиномов
    xNodes2, yNodes2, value2, coeffs2 = buildPolynomial(2, xStar)
    xNodes3, yNodes3, value3, coeffs3 = buildPolynomial(3, xStar)

    print("Узлы для многочлена 2-й степени:", xNodes2)
    print("Узлы для многочлена 3-й степени:", xNodes3)

    # Квадратичный полином (синий)
    x_min2, x_max2 = min(xNodes2), max(xNodes2)
    x_range2 = [x_min2 + i * (x_max2 - x_min2) / 199 for i in range(200)]
    y_range2 = [evaluateNewton(xNodes2, coeffs2, x) for x in x_range2]
    plt.plot(x_range2, y_range2, 'b-', linewidth=2, label='P₂(x) (2-я степень)')

    # Кубический полином (красный)
    x_min3, x_max3 = min(xNodes3), max(xNodes3)
    x_range3 = [x_min3 + i * (x_max3 - x_min3) / 199 for i in range(200)]
    y_range3 = [evaluateNewton(xNodes3, coeffs3, x) for x in x_range3]
    plt.plot(x_range3, y_range3, 'r-', linewidth=2, label='P₃(x) (3-я степень)')

    # Исходные точки
    plt.plot(xs, ys, 'ko', markersize=8, markerfacecolor='white', label='Исходные данные')

    # Точка интерполяции
    y_avg = (value2 + value3) / 2
    plt.plot(xStar, y_avg, 'g+', markersize=15, markeredgewidth=3, label=f'x* = {xStar}')

    # Настройка графика
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Интерполяция Ньютона')
    plt.legend()
    xMin, xMax = -5.0, 4.5
    yMin, yMax = -3.0, 4.0
    plt.axis([xMin, xMax, yMin, yMax])
    plt.tight_layout()
    plt.show()


def print_results():
    """Вывод результатов в консоль"""
    # Получаем данные
    xNodes2, yNodes2, value2, coeffs2 = buildPolynomial(2, xStar)
    xNodes3, yNodes3, value3, coeffs3 = buildPolynomial(3, xStar)

    # Строковые представления полиномов
    poly2Str = newtonPolynomialString(xNodes2, coeffs2)
    poly3Str = newtonPolynomialString(xNodes3, coeffs3)

    # Оценки погрешностей
    maxDeriv2 = estimateMaxDerivative(xNodes2, yNodes2)
    maxDeriv3 = estimateMaxDerivative(xNodes3, yNodes3)
    error2 = estimateError(xNodes2, xStar, maxDeriv2)
    error3 = estimateError(xNodes3, xStar, maxDeriv3)

    # Названия вариантов
    quad2Name = "LeftInterval" if quadraticVariant == 1 else "RightInterval"

    if cubicVariant == 1:
        cubic3Name = "LeftAndTwoRight"
    elif cubicVariant == 2:
        cubic3Name = "Middle"
    else:
        cubic3Name = "RightAndTwoSides"

    print("=" * 80)
    print("ИНТЕРПОЛЯЦИЯ НЬЮТОНА")
    print("=" * 80)

    # Исходные данные
    print("\nИСХОДНЫЕ ДАННЫЕ:")
    print("i:   ", " ".join(f"{i:8d}" for i in range(len(xs))))
    print("xi:  ", " ".join(f"{x:8.2f}" for x in xs))
    print("yi:  ", " ".join(f"{y:8.4f}" for y in ys))
    print(f"Точка интерполяции: x* = {xStar:.3f}")

    # Квадратичный полином
    print("\n" + "=" * 80)
    print(f"МНОГОЧЛЕН НЬЮТОНА 2-Й СТЕПЕНИ (вариант: {quad2Name})")
    print("=" * 80)
    print(f"Узлы: " + " ".join([f"x{i}={x:.2f}" for i, x in enumerate(xNodes2)]))
    print(f"\nP₂(x) = {poly2Str}")
    print(f"\nКоэффициенты разделённых разностей:")
    for i, c in enumerate(coeffs2):
        print(f"  f[x₀,...,x{i}] = {c:.6f}")
    print(f"\nP₂({xStar:.3f}) = {value2:.6f}")
    print(f"Оценка погрешности: ≤ {error2:.3e}")

    # Кубический полином
    print("\n" + "=" * 80)
    print(f"МНОГОЧЛЕН НЬЮТОНА 3-Й СТЕПЕНИ (вариант: {cubic3Name})")
    print("=" * 80)
    print(f"Узлы: " + " ".join([f"x{i}={x:.2f}" for i, x in enumerate(xNodes3)]))
    print(f"\nP₃(x) = {poly3Str}")
    print(f"\nКоэффициенты разделённых разностей:")
    for i, c in enumerate(coeffs3):
        print(f"  f[x₀,...,x{i}] = {c:.6f}")
    print(f"\nP₃({xStar:.3f}) = {value3:.6f}")
    print(f"Оценка погрешности: ≤ {error3:.3e}")

    # Проверка в узловых точках
    print("\n" + "=" * 80)
    print("ПРОВЕРКА В УЗЛОВЫХ ТОЧКАХ")
    print("=" * 80)

    print("\nДля P₂(x):")
    for i in range(len(xNodes2)):
        p = evaluateNewton(xNodes2, coeffs2, xNodes2[i])
        print(f"  x={xNodes2[i]:.2f}: P₂(x)={p:.6f}, y={yNodes2[i]:.4f}")

    print("\nДля P₃(x):")
    for i in range(len(xNodes3)):
        p = evaluateNewton(xNodes3, coeffs3, xNodes3[i])
        print(f"  x={xNodes3[i]:.2f}: P₃(x)={p:.6f}, y={yNodes3[i]:.4f}")

    # Сравнение результатов
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 80)
    print(f"P₂({xStar:.3f}) = {value2:.6f}")
    print(f"P₃({xStar:.3f}) = {value3:.6f}")
    print(f"Разница |P₃ - P₂| = {abs(value3 - value2):.6f}")

    # Информация о вариантах
    print("\n" + "=" * 80)
    print("ВАРИАНТЫ ВЫБОРА УЗЛОВ")
    print("=" * 80)
    print("Квадратичный (quadraticVariant):")
    print("  1 - LeftInterval: x* в левом интервале [i-1, i, i+1]")
    print("  2 - RightInterval: x* в правом интервале [i, i+1, i+2]")
    print("\nКубический (cubicVariant):")
    print("  1 - LeftAndTwoRight: x* слева, 2 справа [i, i+1, i+2, i+3]")
    print("  2 - Middle: x* в среднем [i-1, i, i+1, i+2]")
    print("  3 - RightAndTwoSides: x* справа, 2 по бокам [i-2, i-1, i, i+1]")
    print(f"\nТекущие настройки: quadraticVariant={quadraticVariant}, cubicVariant={cubicVariant}")


def main():
    """Основная функция"""
    print_results()
    create_plot()


if __name__ == "__main__":
    main()