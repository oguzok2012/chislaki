import matplotlib.pyplot as plt
import numpy as np

# Исходные данные
xi = [-4.14, -3.16, -2.18, -1.20, -0.22, 0.76, 1.74, 2.72, 3.70]
yi = [-2.3285, -0.7683, -1.2326, 0.4502, 1.0379, 2.9643, 2.8924, 3.3481, 2.7326]
xStar = 1.493

# Варианты выбора узлов
quadraticVariant = 2  # 1=LeftInterval, 2=RightInterval
cubicVariant = 2  # 1=LeftAndTwoRight, 2=Middle, 3=RightAndTwoSides


def lagrangeBasis(i, x, xNodes):
    """Вычисление базисного полинома Лагранжа"""
    result = 1.0
    for j in range(len(xNodes)):
        if j != i:
            result *= (x - xNodes[j]) / (xNodes[i] - xNodes[j])
    return result


def lagrangePolynomial(x, xNodes, yNodes):
    """Вычисление значения полинома Лагранжа в точке x"""
    result = 0.0
    for i in range(len(xNodes)):
        result += yNodes[i] * lagrangeBasis(i, x, xNodes)
    return result


def lagrangePolynomialString(xNodes, yNodes):
    """Формирование строкового представления полинома Лагранжа"""
    parts = []

    for i in range(len(yNodes)):
        # Числитель базисного многочлена
        numerator = f"{abs(yNodes[i]):.4f}"
        for j in range(len(xNodes)):
            if j != i:
                if xNodes[j] >= 0:
                    numerator += f"(x-{xNodes[j]:.2f})"
                else:
                    numerator += f"(x+{abs(xNodes[j]):.2f})"

        # Знаменатель базисного многочлена
        denominator = ""
        for j in range(len(xNodes)):
            if j != i:
                diff = xNodes[i] - xNodes[j]
                denominator += f"({diff:.2f})"

        # Формируем дробь
        sign = "-" if yNodes[i] < 0 else ""
        max_len = max(len(numerator), len(denominator))
        line = "─" * (max_len - 8)

        fraction = f"{sign}{numerator}\n{line}\n{denominator}"
        parts.append(fraction)

    return "\n\n + \n\n".join(parts)


def buildPolynomial2LeftInterval(xStar):
    """Построение квадратичного полинома - левый интервал"""
    leftIdx = 0
    for i in range(len(xi) - 1):
        if xi[i] <= xStar and xi[i + 1] >= xStar:
            leftIdx = i
            break

    xNodes = [xi[leftIdx - 1], xi[leftIdx], xi[leftIdx + 1]]
    yNodes = [yi[leftIdx - 1], yi[leftIdx], yi[leftIdx + 1]]
    value = lagrangePolynomial(xStar, xNodes, yNodes)
    return xNodes, yNodes, value


def buildPolynomial2RightInterval(xStar):
    """Построение квадратичного полинома - правый интервал"""
    leftIdx = 0
    for i in range(len(xi) - 1):
        if xi[i] <= xStar and xi[i + 1] >= xStar:
            leftIdx = i
            break

    xNodes = [xi[leftIdx], xi[leftIdx + 1], xi[leftIdx + 2]]
    yNodes = [yi[leftIdx], yi[leftIdx + 1], yi[leftIdx + 2]]
    value = lagrangePolynomial(xStar, xNodes, yNodes)
    return xNodes, yNodes, value


def buildPolynomial3LeftAndTwoRight(xStar):
    """Построение кубического полинома - слева и два справа"""
    leftIdx = 0
    for i in range(len(xi) - 1):
        if xi[i] <= xStar and xi[i + 1] >= xStar:
            leftIdx = i
            break

    xNodes = [xi[leftIdx], xi[leftIdx + 1], xi[leftIdx + 2], xi[leftIdx + 3]]
    yNodes = [yi[leftIdx], yi[leftIdx + 1], yi[leftIdx + 2], yi[leftIdx + 3]]
    value = lagrangePolynomial(xStar, xNodes, yNodes)
    return xNodes, yNodes, value


def buildPolynomial3Middle(xStar):
    """Построение кубического полинома - средний вариант"""
    leftIdx = 0
    for i in range(len(xi) - 1):
        if xi[i] <= xStar and xi[i + 1] >= xStar:
            leftIdx = i
            break

    xNodes = [xi[leftIdx - 1], xi[leftIdx], xi[leftIdx + 1], xi[leftIdx + 2]]
    yNodes = [yi[leftIdx - 1], yi[leftIdx], yi[leftIdx + 1], yi[leftIdx + 2]]
    value = lagrangePolynomial(xStar, xNodes, yNodes)
    return xNodes, yNodes, value


def buildPolynomial3RightAndTwoSides(xStar):
    """Построение кубического полинома - справа и два слева"""
    leftIdx = 0
    for i in range(len(xi) - 1):
        if xi[i] <= xStar and xi[i + 1] >= xStar:
            leftIdx = i
            break

    xNodes = [xi[leftIdx - 2], xi[leftIdx - 1], xi[leftIdx], xi[leftIdx + 1]]
    yNodes = [yi[leftIdx - 2], yi[leftIdx - 1], yi[leftIdx], yi[leftIdx + 1]]
    value = lagrangePolynomial(xStar, xNodes, yNodes)
    return xNodes, yNodes, value


def buildPolynomial(degree, xStar):
    """Построение полинома заданной степени"""
    if degree == 2:
        if quadraticVariant == 2:
            return buildPolynomial2RightInterval(xStar)
        return buildPolynomial2LeftInterval(xStar)

    if cubicVariant == 2:
        return buildPolynomial3Middle(xStar)
    elif cubicVariant == 3:
        return buildPolynomial3RightAndTwoSides(xStar)
    else:
        return buildPolynomial3LeftAndTwoRight(xStar)


def create_plot():
    """Создание графика"""
    plt.figure(figsize=(12, 8))

    # Построение полиномов
    xNodes2, yNodes2, value2 = buildPolynomial(2, xStar)
    xNodes3, yNodes3, value3 = buildPolynomial(3, xStar)

    # Квадратичный полином
    x_range2 = np.linspace(min(xNodes2), max(xNodes2), 200)
    y_range2 = [lagrangePolynomial(x, xNodes2, yNodes2) for x in x_range2]
    plt.plot(x_range2, y_range2, 'b-', linewidth=2, label=f'L₂(x) (2-я степень)')

    # Кубический полином
    x_range3 = np.linspace(min(xNodes3), max(xNodes3), 200)
    y_range3 = [lagrangePolynomial(x, xNodes3, yNodes3) for x in x_range3]
    plt.plot(x_range3, y_range3, 'r-', linewidth=2, label=f'L₃(x) (3-я степень)')

    # Исходные точки
    plt.plot(xi, yi, 'ko', markersize=8, markerfacecolor='white', label='Исходные данные')

    # Точка интерполяции
    y_avg = (value2 + value3) / 2
    plt.plot(xStar, y_avg, 'g+', markersize=15, markeredgewidth=3, label=f'x* = {xStar}')

    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Интерполяция Лагранжа')
    plt.legend()
    xMin, xMax = -5.0, 4.5
    yMin, yMax = -3.0, 4.0
    plt.axis([xMin, xMax, yMin, yMax])
    plt.tight_layout()
    plt.show()


def print_results():
    """Вывод результатов в консоль"""
    xNodes2, yNodes2, value2 = buildPolynomial(2, xStar)
    xNodes3, yNodes3, value3 = buildPolynomial(3, xStar)

    poly2Str = lagrangePolynomialString(xNodes2, yNodes2)
    poly3Str = lagrangePolynomialString(xNodes3, yNodes3)

    # Названия вариантов
    quad2Name = "LeftInterval" if quadraticVariant == 1 else "RightInterval"

    if cubicVariant == 1:
        cubic3Name = "LeftAndTwoRight"
    elif cubicVariant == 2:
        cubic3Name = "Middle"
    else:
        cubic3Name = "RightAndTwoSides"

    print("=" * 80)
    print("ИНТЕРПОЛЯЦИЯ ЛАГРАНЖА")
    print("=" * 80)

    # Исходные данные
    print("\nИСХОДНЫЕ ДАННЫЕ:")
    print("i:   ", " ".join(f"{i:8d}" for i in range(len(xi))))
    print("xi:  ", " ".join(f"{x:8.2f}" for x in xi))
    print("yi:  ", " ".join(f"{y:8.4f}" for y in yi))
    print(f"Точка интерполяции: x* = {xStar:.3f}")

    # Квадратичный полином
    print("\n" + "=" * 80)
    print(f"МНОГОЧЛЕН ЛАГРАНЖА 2-Й СТЕПЕНИ (вариант: {quad2Name})")
    print("=" * 80)
    print(f"Узлы: " + " ".join([f"x{i}={x:.2f}" for i, x in enumerate(xNodes2)]))
    print(f"\nL₂(x) = ")
    print(poly2Str)
    print(f"\nL₂({xStar:.3f}) = {value2:.6f}")

    # Кубический полином
    print("\n" + "=" * 80)
    print(f"МНОГОЧЛЕН ЛАГРАНЖА 3-Й СТЕПЕНИ (вариант: {cubic3Name})")
    print("=" * 80)
    print(f"Узлы: " + " ".join([f"x{i}={x:.2f}" for i, x in enumerate(xNodes3)]))
    print(f"\nL₃(x) = ")
    print(poly3Str)
    print(f"\nL₃({xStar:.3f}) = {value3:.6f}")

    # Проверка в узловых точках
    print("\n" + "=" * 80)
    print("ПРОВЕРКА В УЗЛОВЫХ ТОЧКАХ")
    print("=" * 80)

    print("\nДля L₂(x):")
    for i in range(len(xNodes2)):
        p = lagrangePolynomial(xNodes2[i], xNodes2, yNodes2)
        print(f"  x={xNodes2[i]:.2f}: L₂(x)={p:.6f}, y={yNodes2[i]:.4f}")

    print("\nДля L₃(x):")
    for i in range(len(xNodes3)):
        p = lagrangePolynomial(xNodes3[i], xNodes3, yNodes3)
        print(f"  x={xNodes3[i]:.2f}: L₃(x)={p:.6f}, y={yNodes3[i]:.4f}")

    # Сравнение результатов
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 80)
    print(f"L₂({xStar:.3f}) = {value2:.6f}")
    print(f"L₃({xStar:.3f}) = {value3:.6f}")
    print(f"Разница |L₃ - L₂| = {abs(value3 - value2):.6f}")

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
    print("  3 - RightAndTwoSides: x* справа, 2 по бокам [i-2, i-1, i, i+3]")
    print(f"\nТекущие настройки: quadraticVariant={quadraticVariant}, cubicVariant={cubicVariant}")


def main():
    """Основная функция"""
    print_results()
    create_plot()


if __name__ == "__main__":
    main()