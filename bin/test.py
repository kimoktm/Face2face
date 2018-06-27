# import numpy as np
import autograd.numpy as np
from autograd import value_and_grad
from autograd import grad
from scipy.optimize import minimize
import time


x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
y0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])


def objective(x, y):
	f = np.tensordot(x, y, axes = 0)
	return np.sum(f)

def rosen(x, y, z):
    """The Rosenbrock function"""
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0) * y + z

def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der


# print(rosen(x0, 2, 1))
# print(objective(x0, y0))


x_a = np.c_[[x0, x0 * 2]]
xxx = grad(rosen)
print(xxx(x_a, 2, 1))

# res = minimize(rosen, x0, method='BFGS', jac=rosen_der,
#                options={'disp': True})

res = minimize(value_and_grad(rosen), x_a, args = (10, 1), method='BFGS', jac=True,
               options={'disp': True})


print(res.x)