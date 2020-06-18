#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
http://www.scipy.org/Cookbook/Least_Squares_Circle
"""

from numpy import *

# Coordinates of the 2D points

# x = r_[  9, 35, -13,  10,  23,   0]
# y = r_[ 34, 10,   6, -14,  27, -10]

# x = r_[36, 36, 19, 18, 33, 26]
# y = r_[14, 10, 28, 31, 18, 26]
x = r_[68.2533, 74.9322, 82.4288, 104.337, 111.892, 127.953, 142.97, 156.818, 175.432, 184.949]
y = r_[98.9417, 132.844, 67.0712, 148.646, 50.9481, 159.973, 45.1403, 166.788, 51.6778, 152.553]

# R0 = 25
# nb_pts = 8
# dR = 2
# angle =9*pi/5
# x = (10 + R0*cos(theta0) + dR*random.normal(size=nb_pts)).round()
# y = (10 + R0*sin(theta0) + dR*random.normal(size=nb_pts)).round()


# == METHOD 1 ==
method_1 = 'algebraic'

# coordinates of the barycenter
x_m = mean(x)
y_m = mean(y)

# calculation of the reduced coordinates
u = x - x_m
v = y - y_m

# linear system defining the center in reduced coordinates (uc, vc):
#    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
#    Suv * uc +  Svv * vc = (Suuv + Svvv)/2

Suv = sum(u * v)
Suu = sum(u ** 2)
Svv = sum(v ** 2)
Suuv = sum(u ** 2 * v)
Suvv = sum(u * v ** 2)
Suuu = sum(u ** 3)
Svvv = sum(v ** 3)

# Solving the linear system
A = array([[Suu, Suv], [Suv, Svv]])
B = array([Suuu + Suvv, Svvv + Suuv]) / 2.0
uc, vc = linalg.solve(A, B)

xc_1 = x_m + uc
yc_1 = y_m + vc

# Calculation of all distances from the center (xc_1, yc_1)
Ri_1 = sqrt((x - xc_1) ** 2 + (y - yc_1) ** 2)
R_1 = mean(Ri_1)
residu_1 = sum((Ri_1 - R_1) ** 2)
residu2_1 = sum((Ri_1 ** 2 - R_1 ** 2) ** 2)

# Decorator to count functions calls
import functools


def countcalls(fn):
    "decorator function count function calls "

    @functools.wraps(fn)
    def wrapped(*args):
        wrapped.ncalls += 1
        return fn(*args)

    wrapped.ncalls = 0
    return wrapped


#  == METHOD 2 ==
from scipy import optimize

method_2 = "leastsq"


def calc_R(c):
    """ calculate the distance of each 2D points from the center c=(xc, yc) """
    return sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2)


@countcalls
def f_2(c):
    """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(c)
    return Ri - Ri.mean()


center_estimate = x_m, y_m
center_2, ier = optimize.leastsq(f_2, center_estimate)

xc_2, yc_2 = center_2
Ri_2 = calc_R(center_2)
R_2 = Ri_2.mean()
residu_2 = sum((Ri_2 - R_2) ** 2)
residu2_2 = sum((Ri_2 ** 2 - R_2 ** 2) ** 2)
ncalls_2 = f_2.ncalls

# == METHOD 3 ==
from scipy import odr

method_3 = "odr"


@countcalls
def f_3(beta, x):
    """ implicit function of the circle """
    xc, yc, r = beta
    return (x[0] - xc) ** 2 + (x[1] - yc) ** 2 - r ** 2


def calc_estimate(data):
    """ Return a first estimation on the parameter from the data  """
    xc0, yc0 = data.x.mean(axis=1)
    r0 = sqrt((data.x[0] - xc0) ** 2 + (data.x[1] - yc0) ** 2).mean()
    return xc0, yc0, r0


# for implicit function :
#       data.x contains both coordinates of the points
#       data.y is the dimensionality of the response
lsc_data = odr.Data(row_stack([x, y]), y=1)
lsc_model = odr.Model(f_3, implicit=True, estimate=calc_estimate)
lsc_odr = odr.ODR(lsc_data, lsc_model)
lsc_out = lsc_odr.run()

xc_3, yc_3, R_3 = lsc_out.beta
Ri_3 = calc_R([xc_3, yc_3])
residu_3 = sum((Ri_3 - R_3) ** 2)
residu2_3 = sum((Ri_3 ** 2 - R_3 ** 2) ** 2)
ncalls_3 = f_3.ncalls

print('lsc_out.sum_square = ', lsc_out.sum_square)

# == METHOD 4 ==

method_4 = "odr with jacobian"


@countcalls
def f_4(beta, x):
    """ implicit function of the circle """
    xc, yc, r = beta
    xi, yi = x

    return (xi - xc) ** 2 + (yi - yc) ** 2 - r ** 2


@countcalls
def jacb(beta, x):
    """ Jacobian function with respect to the parameters beta.
    return df/dbeta
    """
    xc, yc, r = beta
    xi, yi = x

    df_db = empty((beta.size, x.shape[1]))
    df_db[0] = 2 * (xc - xi)  # d_f/dxc
    df_db[1] = 2 * (yc - yi)  # d_f/dyc
    df_db[2] = -2 * r  # d_f/dr

    return df_db


@countcalls
def jacd(beta, x):
    """ Jacobian function with respect to the input x.
    return df/dx
    """
    xc, yc, r = beta
    xi, yi = x

    df_dx = empty_like(x)
    df_dx[0] = 2 * (xi - xc)  # d_f/dxi
    df_dx[1] = 2 * (yi - yc)  # d_f/dyi

    return df_dx


def calc_estimate(data):
    """ Return a first estimation on the parameter from the data  """
    xc0, yc0 = data.x.mean(axis=1)
    r0 = sqrt((data.x[0] - xc0) ** 2 + (data.x[1] - yc0) ** 2).mean()
    return xc0, yc0, r0


# for implicit function :
#       data.x contains both coordinates of the points
#       data.y is the dimensionality of the response
lsc_data = odr.Data(row_stack([x, y]), y=1)
lsc_model = odr.Model(f_4, implicit=True, estimate=calc_estimate, fjacd=jacd, fjacb=jacb)
lsc_odr = odr.ODR(lsc_data, lsc_model)
lsc_odr.set_job(deriv=3)  # use user derivatives function without checking
lsc_out = lsc_odr.run()

xc_4, yc_4, R_4 = lsc_out.beta
Ri_4 = calc_R([xc_4, yc_4])
residu_4 = sum((Ri_4 - R_4) ** 2)
residu2_4 = sum((Ri_4 ** 2 - R_4 ** 2) ** 2)
ncalls_4 = f_4.ncalls

print("Method 4 :")
print("Functions calls : f_4=%d jacb=%d jacd=%d" % (f_4.ncalls, jacb.ncalls, jacd.ncalls))

# Summary
fmt = '%-18s %10.5f %10.5f %10.5f %10d %10.6f %10.6f %10.2f'
# print('\n%-18s' + ' %10s' * 7) % tuple('METHOD Xc Yc Rc nb_calls std(Ri) residu residu2'.split())
print('-' * (18 + 7 * (10 + 1)))
print(fmt % (method_1, xc_1, yc_1, R_1, 1, Ri_1.std(), residu_1, residu2_1))
print(fmt % (method_2, xc_2, yc_2, R_2, ncalls_2, Ri_2.std(), residu_2, residu2_2))
print(fmt % (method_3, xc_3, yc_3, R_3, ncalls_3, Ri_3.std(), residu_3, residu2_3))
print(fmt % (method_4, xc_4, yc_4, R_4, ncalls_4, Ri_4.std(), residu_4, residu2_4))

# plotting functions

from matplotlib import pyplot as p, cm


def plot_all(residu2=False, basename='circle'):
    """ Draw data points, best fit circles and center for the three methods,
    and adds the iso contours corresponding to the fiel residu or residu2
    """

    f = p.figure(figsize=(6.5, 4.5), dpi=90, facecolor='white')
    p.axis('equal')

    p.plot(x, y, 'ro', label='data', ms=9, mec='b', mew=1)

    theta_fit = linspace(-pi, pi, 180)

    x_fit1 = xc_1 + R_1 * cos(theta_fit)
    y_fit1 = yc_1 + R_1 * sin(theta_fit)
    p.plot(x_fit1, y_fit1, 'b-', label=method_1, lw=2)

    x_fit2 = xc_2 + R_2 * cos(theta_fit)
    y_fit2 = yc_2 + R_2 * sin(theta_fit)
    p.plot(x_fit2, y_fit2, 'k--', label=method_2, lw=2)

    x_fit3 = xc_3 + R_3 * cos(theta_fit)
    y_fit3 = yc_3 + R_3 * sin(theta_fit)
    p.plot(x_fit3, y_fit3, 'r-.', label=method_3, lw=2)

    p.plot([xc_1], [yc_1], 'bD', mec='y', mew=1)
    p.plot([xc_2], [yc_2], 'gD', mec='r', mew=1)
    p.plot([xc_3], [yc_3], 'kD', mec='w', mew=1)

    # draw
    p.xlabel('x')
    p.ylabel('y')
    p.legend(loc='best', labelspacing=0.1)

    # plot the residu fields
    nb_pts = 100

    p.draw()
    xmin, xmax = p.xlim()
    ymin, ymax = p.ylim()

    vmin = min(xmin, ymin)
    vmax = max(xmax, ymax)

    xg, yg = ogrid[vmin:vmax:nb_pts * 1j, vmin:vmax:nb_pts * 1j]
    xg = xg[..., newaxis]
    yg = yg[..., newaxis]

    Rig = sqrt((xg - x) ** 2 + (yg - y) ** 2)
    Rig_m = Rig.mean(axis=2)[..., newaxis]

    if residu2:
        residu = sum((Rig ** 2 - Rig_m ** 2) ** 2, axis=2)
    else:
        residu = sum((Rig - Rig_m) ** 2, axis=2)

    lvl = exp(linspace(log(residu.min()), log(residu.max()), 15))

    p.contourf(xg.flat, yg.flat, residu.T, lvl, alpha=0.75, cmap=cm.Purples_r)
    cbar = p.colorbar(format='%.f')

    if residu2:
        cbar.set_label('Residu_2')
    else:
        cbar.set_label('Residu')

    p.xlim(xmin=vmin, xmax=vmax)
    p.ylim(ymin=vmin, ymax=vmax)

    p.grid()
    p.title('Leasts Squares Circle')
    p.savefig('%s_residu%d.png' % (basename, 2 if residu2 else 1))


plot_all(residu2=False, basename='circle')
plot_all(residu2=True, basename='circle')

p.show()
# vim: set et sts=4 sw=4:
