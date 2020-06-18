from scipy import optimize
from scipy.optimize import least_squares
import numpy as np

xdata = np.linspace(-10, 10, num=20)


def f2(x, A, B, R):
    return np.sqrt(np.abs(R*R - (x - A)*(x - A))) + B


guess = [0, 0, 12]
ydata = f2(xdata, guess[0], guess[1], guess[2]) + 0.1 * np.random.randn(xdata.size)
params, params_covariance = optimize.curve_fit(f2, xdata, ydata, guess)
print(params)
