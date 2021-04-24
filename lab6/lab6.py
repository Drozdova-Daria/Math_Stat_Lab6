import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def linear_regression(x, a, b):
    Y = []
    for x in X:
        Y.append(a + b * x )

    return Y


def least_squares_method_coefficients(X, Y):
    x_ = np.mean(X)
    y_ = np.mean(Y)
    x2_ = np.mean(X * X)
    xy_ = np.mean(X * Y)
    beta_1 = (xy_ - x_ * y_) / (x2_ - (x_) ** 2)
    beta_0 = (y_ - x_ * beta_1)

    return beta_0, beta_1


def least_modules_function(beta, X, Y):
    sum = 0
    for x, y in zip(X, Y):
        sum += abs(y - beta[0] - beta[1] * x)

    return sum


def least_modules_method_coefficients(X, Y):
    beta_0, beta_1 = least_squares_method_coefficients(X, Y)
    result = minimize(least_modules_function, [beta_0, beta_1], args=(X, Y), method='SLSQP')

    return result.x[0], result.x[1]


def get_ratings(X, Y):
    beta_0_ls, beta_1_ls = least_squares_method_coefficients(X, Y)
    print('Критерий наименьших квадратов:')
    print('a = ' + str(round(beta_0_ls, 5)) + ', b = ' + str(round(beta_1_ls, 5)))
    Y_ls = linear_regression(X, beta_0_ls, beta_1_ls)
    beta_0_lm, beta_1_lm = least_modules_method_coefficients(X, Y)
    print('Критерий наименьших модулей:')
    print('a = ' + str(round(beta_0_lm, 5)) + ', b = ' + str(round(beta_1_lm, 5)))
    Y_lm = linear_regression(X, beta_0_lm, beta_1_lm)

    return Y_ls, Y_lm


def draw_image(X, Y, Y_e, Y_ls, Y_lm, name):
    plt.scatter(X, Y_e, label='Выборка', color='red')
    plt.plot(X, Y, label='Модель')
    plt.plot(X, Y_ls, label='МНК')
    plt.plot(X, Y_lm, label='МНМ')

    plt.title('Оценки коэффициентов линейной регрессии')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(name + '.png')
    plt.show()


if __name__ == '__main__':
    a = -1.8
    b = 2
    step = 0.2

    a_ref = 2
    b_ref = 2

    outrage_1 = 10
    outrage_20 = -10

    X = np.arange(a, b + step, step)
    Y = linear_regression(X, a_ref, b_ref)
    Y_e = [(y + stats.norm.rvs(0, 1)) for y in Y]
    print('Выборка без возмущения')
    Y_ls, Y_lm = get_ratings(X, Y_e)
    draw_image(X, Y, Y_e, Y_ls, Y_lm, '1')

    Y_e[0] += outrage_1
    Y_e[-1] += outrage_20
    print('Выборка с возмущнием')
    Y_ls, Y_lm = get_ratings(X, Y_e)
    draw_image(X, Y, Y_e, Y_ls, Y_lm, '2')
