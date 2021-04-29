import numpy as np
from numpy.random import normal
import pandas as pd

from initialization import preprocess_data, supp_warm_start

def cost(x, y, s, k, gamma):
  alpha = alpha_prime(x, y, s, k, gamma)
  result = 0.5 * gamma * y.T.dot(alpha)
  return result

def alpha_prime(x, y, s, k, gamma):
  curr_index = np.nonzero(s)[0]
  x_s_t = x[:, curr_index].T
  x_s = x[:, curr_index]
  curr_kernel = np.dot(x_s_t, x_s)
  inner_calc = x_s.dot(np.linalg.inv( (np.identity(k)/gamma) + curr_kernel)).dot(x_s_t)
  result = y - inner_calc.dot(y)
  return result

def gradient_s_i(x, y, s, k, gamma):
  gradient_c = []
  for j in range(0, np.shape(x)[1]):
    gradient_c.append(-0.5 * gamma * (x[:, j].T.dot(alpha_prime(x, y, s, k, gamma))**2 ))
  return np.array(gradient_c)

def support_approx(x, y, curr_s, k, gamma, supp_size):
  eta = 0
  gradient = 0
  counter = 1
  while eta < cost(x, y, curr_s, k, gamma) and counter < supp_size:
    print('Iteration {}:'.format(counter))
    gradient = gradient_s_i(x, y, curr_s, k, gamma)
    k += 1
    s_bar = np.zeros(np.shape(x)[1])
    s_bar[np.argsort(gradient)[:k]] = 1
    eta = cost(x, y, curr_s, k-1, gamma) + gradient.T.dot(s_bar - curr_s)
    print("Current eta: {}".format(eta))
    curr_s = s_bar
    print("Current cost: {}".format(cost(x, y, curr_s, k, gamma)))
    counter += 1
    print('')
  return curr_s, k 

'''
Run the iteration and check the cost and sub gradient, 
and calculate the beta and support for the current matrix model
'''
def solve(x, y, supp_size, gamma):
  k = 1
  x_centered, y_centered, mean_x, mean_y, x_l2_norm_square, beta_multiplier = preprocess_data(x, y, True, True)
  warm_start_s = supp_warm_start(x_centered, y_centered, x_l2_norm_square)
  s, k = support_approx(x_centered, y_centered, warm_start_s, k, gamma, supp_size)
  x_s = np.zeros(np.shape(x_centered))
  x_s[:, np.nonzero(s)[0]] = x_centered[:, np.nonzero(s)[0]]
  beta = np.linalg.inv((np.identity(np.shape(x_centered)[1])/gamma) + (x_s.T.dot(x_s))).dot(x_s.T).dot(y_centered)
  return beta, s