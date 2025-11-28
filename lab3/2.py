import math
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button

# Исходные данные
xs = [-4.14, -3.16, -2.18, -1.20, -0.22, 0.76, 1.74, 2.72, 3.70]
ys = [-2.3285, -0.7683, -1.2326, 0.4502, 1.0379, 2.9643, 2.8924, 3.3481, 2.7326]
xStar = 1.493


class NewtonInterpolation:
    def __init__(self, xs, ys, xStar):
        self.xs = xs
        self.ys = ys
        self.xStar = xStar

    def dividedDifferences(self, X, Y):
        """Вычисление разделенных разностей"""
        n = len(X)
        table = [[0.0] * n for _ in range(n)]

        for i in range(n):
            table[i][0] = Y[i]

        for j in range(1, n):
            for i in range(n - j):
                table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (X[i + j] - X[i])

        return table[0]

    def evaluateNewton(self, X, coeffs, x):
        """Вычисление значения полинома Ньютона в точке x"""
        result = coeffs[0]
        product = 1.0
        for i in range(1, len(coeffs)):
            product *= (x - X[i - 1])
            result += coeffs[i] * product
        return result

    def buildPolynomial2LeftInterval(self, xStar):
        """Построение квадратичного полинома - левый интервал"""
        leftIdx = 0
        for i in range(len(self.xs) - 1):
            if self.xs[i] <= xStar and self.xs[i + 1] >= xStar:
                leftIdx = i
                break

        xNodes = [self.xs[leftIdx - 1], self.xs[leftIdx], self.xs[leftIdx + 1]]
        yNodes = [self.ys[leftIdx - 1], self.ys[leftIdx], self.ys[leftIdx + 1]]
        coeffs = self.dividedDifferences(xNodes, yNodes)
        value = self.evaluateNewton(xNodes, coeffs, xStar)
        return xNodes, yNodes, value, coeffs

    def buildPolynomial2RightInterval(self, xStar):
        """Построение квадратичного полинома - правый интервал"""
        leftIdx = 0
        for i in range(len(self.xs) - 1):
            if self.xs[i] <= xStar and self.xs[i + 1] >= xStar:
                leftIdx = i
                break

        xNodes = [self.xs[leftIdx], self.xs[leftIdx + 1], self.xs[leftIdx + 2]]
        yNodes = [self.ys[leftIdx], self.ys[leftIdx + 1], self.ys[leftIdx + 2]]
        coeffs = self.dividedDifferences(xNodes, yNodes)
        value = self.evaluateNewton(xNodes, coeffs, xStar)
        return xNodes, yNodes, value, coeffs

    def buildPolynomial3LeftAndTwoRight(self, xStar):
        """Построение кубического полинома - слева и два справа"""
        leftIdx = 0
        for i in range(len(self.xs) - 1):
            if self.xs[i] <= xStar and self.xs[i + 1] >= xStar:
                leftIdx = i
                break

        xNodes = [self.xs[leftIdx], self.xs[leftIdx + 1], self.xs[leftIdx + 2], self.xs[leftIdx + 3]]
        yNodes = [self.ys[leftIdx], self.ys[leftIdx + 1], self.ys[leftIdx + 2], self.ys[leftIdx + 3]]
        coeffs = self.dividedDifferences(xNodes, yNodes)
        value = self.evaluateNewton(xNodes, coeffs, xStar)
        return xNodes, yNodes, value, coeffs

    def buildPolynomial3Middle(self, xStar):
        """Построение кубического полинома - средний вариант"""
        leftIdx = 0
        for i in range(len(self.xs) - 1):
            if self.xs[i] <= xStar and self.xs[i + 1] >= xStar:
                leftIdx = i
                break

        xNodes = [self.xs[leftIdx - 1], self.xs[leftIdx], self.xs[leftIdx + 1], self.xs[leftIdx + 2]]
        yNodes = [self.ys[leftIdx - 1], self.ys[leftIdx], self.ys[leftIdx + 1], self.ys[leftIdx + 2]]
        coeffs = self.dividedDifferences(xNodes, yNodes)
        value = self.evaluateNewton(xNodes, coeffs, xStar)
        return xNodes, yNodes, value, coeffs

    def buildPolynomial3RightAndTwoSides(self, xStar):
        """Построение кубического полинома - справа и два по бокам"""
        leftIdx = 0
        for i in range(len(self.xs) - 1):
            if self.xs[i] <= xStar and self.xs[i + 1] >= xStar:
                leftIdx = i
                break

        xNodes = [self.xs[leftIdx - 2], self.xs[leftIdx - 1], self.xs[leftIdx], self.xs[leftIdx + 1]]
        yNodes = [self.ys[leftIdx - 2], self.ys[leftIdx - 1], self.ys[leftIdx], self.ys[leftIdx + 1]]
        coeffs = self.dividedDifferences(xNodes, yNodes)
        value = self.evaluateNewton(xNodes, coeffs, xStar)
        return xNodes, yNodes, value, coeffs

    def buildPolynomial(self, degree, quadraticVariant, cubicVariant, xStar):
        """Построение полинома заданной степени"""
        if degree == 2:
            if quadraticVariant == 2:
                xNodes, yNodes, value, coeffs = self.buildPolynomial2RightInterval(xStar)
            else:
                xNodes, yNodes, value, coeffs = self.buildPolynomial2LeftInterval(xStar)
        else:
            if cubicVariant == 2:
                xNodes, yNodes, value, coeffs = self.buildPolynomial3Middle(xStar)
            elif cubicVariant == 3:
                xNodes, yNodes, value, coeffs = self.buildPolynomial3RightAndTwoSides(xStar)
            else:
                xNodes, yNodes, value, coeffs = self.buildPolynomial3LeftAndTwoRight(xStar)

        return xNodes, yNodes, value, coeffs

    def newtonPolynomialString(self, X, coeffs):
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

    def estimateMaxDerivative(self, X, Y):
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

    def estimateError(self, X, x, maxDerivative):
        """Оценка погрешности интерполяции"""
        omega = 1.0
        for i in range(len(X)):
            omega *= abs(x - X[i])

        n = len(X) - 1
        factorial = 1.0
        for i in range(2, n + 2):
            factorial *= i

        return (maxDerivative / factorial) * omega


def print_all_results():
    """Вывод результатов для всех комбинаций вариантов"""
    interpolator = NewtonInterpolation(xs, ys, xStar)

    quadratic_variants = [
        (1, "LeftInterval: x* в левом интервале [i-1, i, i+1]"),
        (2, "RightInterval: x* в правом интервале [i, i+1, i+2]")
    ]

    cubic_variants = [
        (1, "LeftAndTwoRight: x* слева, 2 справа [i, i+1, i+2, i+3]"),
        (2, "Middle: x* в среднем [i-1, i, i+1, i+2]"),
        (3, "RightAndTwoSides: x* справа, 2 по бокам [i-2, i-1, i, i+1]")
    ]

    print("=" * 100)
    print("ИНТЕРПОЛЯЦИЯ НЬЮТОНА - ВСЕ ВАРИАНТЫ")
    print("=" * 100)

    # Исходные данные
    print("\nИСХОДНЫЕ ДАННЫЕ:")
    print("i:   ", " ".join(f"{i:8d}" for i in range(len(xs))))
    print("xi:  ", " ".join(f"{x:8.2f}" for x in xs))
    print("yi:  ", " ".join(f"{y:8.4f}" for y in ys))
    print(f"Точка интерполяции: x* = {xStar:.3f}")

    # Перебор всех комбинаций
    for quad_var, quad_desc in quadratic_variants:
        for cub_var, cub_desc in cubic_variants:
            print("\n" + "=" * 100)
            print(f"КОМБИНАЦИЯ: quadraticVariant={quad_var}, cubicVariant={cub_var}")
            print("=" * 100)

            # Получаем данные для полиномов
            xNodes2, yNodes2, value2, coeffs2 = interpolator.buildPolynomial(2, quad_var, cub_var, xStar)
            xNodes3, yNodes3, value3, coeffs3 = interpolator.buildPolynomial(3, quad_var, cub_var, xStar)

            # Строковые представления полиномов
            poly2Str = interpolator.newtonPolynomialString(xNodes2, coeffs2)
            poly3Str = interpolator.newtonPolynomialString(xNodes3, coeffs3)

            # Оценки погрешностей
            maxDeriv2 = interpolator.estimateMaxDerivative(xNodes2, yNodes2)
            maxDeriv3 = interpolator.estimateMaxDerivative(xNodes3, yNodes3)
            error2 = interpolator.estimateError(xNodes2, xStar, maxDeriv2)
            error3 = interpolator.estimateError(xNodes3, xStar, maxDeriv3)

            # Квадратичный полином
            print(f"\nМНОГОЧЛЕН НЬЮТОНА 2-Й СТЕПЕНИ ({quad_desc})")
            print(f"Узлы: " + " ".join([f"x{i}={x:.2f}" for i, x in enumerate(xNodes2)]))
            print(f"P₂(x) = {poly2Str}")
            print(f"P₂({xStar:.3f}) = {value2:.6f}")
            print(f"Оценка погрешности: ≤ {error2:.3e}")

            # Кубический полином
            print(f"\nМНОГОЧЛЕН НЬЮТОНА 3-Й СТЕПЕНИ ({cub_desc})")
            print(f"Узлы: " + " ".join([f"x{i}={x:.2f}" for i, x in enumerate(xNodes3)]))
            print(f"P₃(x) = {poly3Str}")
            print(f"P₃({xStar:.3f}) = {value3:.6f}")
            print(f"Оценка погрешности: ≤ {error3:.3e}")

            # Сравнение результатов
            print(f"\nСРАВНЕНИЕ:")
            print(f"P₂({xStar:.3f}) = {value2:.6f}")
            print(f"P₃({xStar:.3f}) = {value3:.6f}")
            print(f"Разница |P₃ - P₂| = {abs(value3 - value2):.6f}")


class InteractivePlot:
    def __init__(self, xs, ys, xStar):
        self.xs = xs
        self.ys = ys
        self.xStar = xStar
        self.interpolator = NewtonInterpolation(xs, ys, xStar)

        self.quadraticVariant = 1
        self.cubicVariant = 2

        self.fig = None
        self.ax = None
        self.quad_radio = None
        self.cubic_radio = None

    def create_interactive_plot(self):
        """Создание интерактивного графика с выбором вариантов"""
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)

        # Создаем области для радио-кнопок
        quad_ax = plt.axes([0.76, 0.7, 0.2, 0.2])
        cubic_ax = plt.axes([0.76, 0.4, 0.2, 0.2])
        update_ax = plt.axes([0.76, 0.1, 0.2, 0.05])

        # Создаем радио-кнопки
        self.quad_radio = RadioButtons(quad_ax, ['LeftInterval\n[i-1,i,i+1]', 'RightInterval\n[i,i+1,i+2]'])
        self.cubic_radio = RadioButtons(cubic_ax, ['LeftAndTwoRight\n[i,i+1,i+2,i+3]',
                                                   'Middle\n[i-1,i,i+1,i+2]',
                                                   'RightAndTwoSides\n[i-2,i-1,i,i+1]'])

        # Кнопка обновления
        update_button = Button(update_ax, 'Обновить график')
        update_button.on_clicked(self.update_plot)

        # Устанавливаем начальные значения
        self.quad_radio.set_active(0)  # LeftInterval
        self.cubic_radio.set_active(1)  # Middle

        # Добавляем подписи
        quad_ax.set_title('Квадратичный полином', fontsize=10)
        cubic_ax.set_title('Кубический полином', fontsize=10)

        # Первоначальное построение графика
        self.update_plot(None)

        plt.show()

    def update_plot(self, event):
        """Обновление графика при изменении выбора"""
        # Получаем текущие значения из радио-кнопок
        quad_active = self.quad_radio.value_selected
        cubic_active = self.cubic_radio.value_selected

        # Определяем числовые значения для вариантов
        quad_options = {
            'LeftInterval\n[i-1,i,i+1]': 1,
            'RightInterval\n[i,i+1,i+2]': 2
        }

        cubic_options = {
            'LeftAndTwoRight\n[i,i+1,i+2,i+3]': 1,
            'Middle\n[i-1,i,i+1,i+2]': 2,
            'RightAndTwoSides\n[i-2,i-1,i,i+1]': 3
        }

        self.quadraticVariant = quad_options[quad_active]
        self.cubicVariant = cubic_options[cubic_active]

        # Очищаем график
        self.ax.clear()

        # Получаем данные для полиномов
        xNodes2, yNodes2, value2, coeffs2 = self.interpolator.buildPolynomial(2, self.quadraticVariant,
                                                                              self.cubicVariant, self.xStar)
        xNodes3, yNodes3, value3, coeffs3 = self.interpolator.buildPolynomial(3, self.quadraticVariant,
                                                                              self.cubicVariant, self.xStar)

        # Названия вариантов
        quad_names = {1: "LeftInterval", 2: "RightInterval"}
        cubic_names = {1: "LeftAndTwoRight", 2: "Middle", 3: "RightAndTwoSides"}

        quad_name = quad_names[self.quadraticVariant]
        cubic_name = cubic_names[self.cubicVariant]

        print(f"\nТекущая комбинация: quadraticVariant={self.quadraticVariant} ({quad_name}), "
              f"cubicVariant={self.cubicVariant} ({cubic_name})")

        # Квадратичный полином (синий)
        x_min2, x_max2 = min(xNodes2), max(xNodes2)
        x_range2 = [x_min2 + i * (x_max2 - x_min2) / 199 for i in range(200)]
        y_range2 = [self.interpolator.evaluateNewton(xNodes2, coeffs2, x) for x in x_range2]
        self.ax.plot(x_range2, y_range2, 'b-', linewidth=2, label=f'P₂(x) ({quad_name})')

        # Кубический полином (красный)
        x_min3, x_max3 = min(xNodes3), max(xNodes3)
        x_range3 = [x_min3 + i * (x_max3 - x_min3) / 199 for i in range(200)]
        y_range3 = [self.interpolator.evaluateNewton(xNodes3, coeffs3, x) for x in x_range3]
        self.ax.plot(x_range3, y_range3, 'r-', linewidth=2, label=f'P₃(x) ({cubic_name})')

        # Исходные точки
        self.ax.plot(self.xs, self.ys, 'ko', markersize=8, markerfacecolor='white', label='Исходные данные')

        # Узлы интерполяции
        self.ax.plot(xNodes2, yNodes2, 'bo', markersize=6, label='Узлы P₂(x)')
        self.ax.plot(xNodes3, yNodes3, 'ro', markersize=6, label='Узлы P₃(x)')

        # Точка интерполяции
        self.ax.plot(self.xStar, value2, 'g+', markersize=15, markeredgewidth=3, label=f'P₂(x*) = {value2:.4f}')
        self.ax.plot(self.xStar, value3, 'm+', markersize=15, markeredgewidth=3, label=f'P₃(x*) = {value3:.4f}')

        # Настройка графика
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title(f'Интерполяция Ньютона\nP₂: {quad_name}, P₃: {cubic_name}')
        self.ax.legend(loc='upper left', bbox_to_anchor=(0, 1))

        xMin, xMax = -5.0, 4.5
        yMin, yMax = -3.0, 4.0
        self.ax.set_xlim(xMin, xMax)
        self.ax.set_ylim(yMin, yMax)

        self.fig.canvas.draw()


def main():
    """Основная функция"""
    print_all_results()

    # Создаем интерактивный график
    plotter = InteractivePlot(xs, ys, xStar)
    plotter.create_interactive_plot()


if __name__ == "__main__":
    main()