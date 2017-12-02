import numpy as np


def normal_equation(x_arr, y_arr):
    """
    This function returns best-fit theta of equation x_arr * theta = y_arr
    Use it if x_arr is not very large matrix
    """
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    x_t_x = x_mat.T * x_mat
    if np.linalg.det(x_t_x) == 0.0:
        print("This matrix is singular")
        return
    theta = x_t_x.I * (x_mat.T * y_mat)
    return theta


def residual(theta, x_arr, y_arr):
    """
    This function returns residual that is calculated as:
    (x_arr * theta - y_arr).T * (x_arr * theta - y_arr)
    """
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr)
    if y_mat.shape[0] == 1:
        y_mat = y_mat.T
    theta_mat = np.mat(theta)
    if theta_mat.shape[0] == 1:
        theta_mat = theta_mat.T
    return float(sum(np.square(x_mat * theta_mat - y_mat)))


def gradient(func, point, x_arr, y_arr, h=0.0001):
    """
    This function returns gradient of function func at point
    Should work for any function with point, x_arr, y_arr parameters
    """
    grad = np.zeros(point.shape[0])
    point_mat = np.mat(point)
    if point_mat.shape[0] == 1:
        point_mat = point_mat.T
    for i in range(0, point_mat.shape[0]):
        point_mat[i] -= h
        f1 = func(point_mat)
        point_mat[i] += 2 * h
        f2 = func(point_mat, x_arr, y_arr)
        grad[i] = ((f2 - f1) / (2 * h))
        point_mat[i] -= h
    return grad

def square(vector):
    """
    Returns the sum of all coordinates
    """
    vector = np.reshape(vector, (-1, 1))
    return float(np.dot(vector.T, vector))

def gradient_descent(theta, x_arr, y_arr, max_iterations=1000, eps=1.e-6):
    """
    This function returns best-fit theta of equation x_arr * theta = y_arr
    Use it if x_arr is large matrix
    Works only for linear regression
    Realization of gradient descent for linear regression
    """
    x = np.array(x_arr)
    y = np.reshape(y_arr, (-1, 1))
    theta = np.reshape(theta, (-1, 1))
    iter = 0
    delta = 1000
    while (iter <= max_iterations) and (delta >= eps):
        prev_theta = np.array(theta)
        resid = np.dot(x, theta) - y
        d_theta = np.array([])
        for i in x.T:
            d_theta = np.append(d_theta, np.dot(i, resid))
        d_theta = np.reshape(d_theta, (-1, 1))

        while (square(np.dot(x, theta) - y) >= square(np.dot(x, (theta - d_theta)) - y)):
            theta -= d_theta
            d_theta *= 2

        while(square(np.dot(x , theta) - y) <= square(np.dot(x , (theta - d_theta)) - y)):
            d_theta *= 0.5

        theta -= d_theta
        delta = float(sum(np.square(prev_theta - theta)))
        iter += 1
        print("Iteration ", iter , "\ntheta:\n", theta, "\n d_theta =\n", d_theta, "epsilon = ", delta)
    return theta



