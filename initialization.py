import numpy as np
from numpy.random import normal
import pandas as pd

'''
Generate synthetic data with number of samples (n) and features (p), along with maximum number of support size and correlation
'''
def generate_synthetic_data(n, p, supp_size=10, snr=10, corr=0, seed=1 ):
  np.random.seed(seed)
  b = np.zeros(p)
  support = []
  for i in range(supp_size):
    support.append(int(i * (p / supp_size)))
  x = normal(size=(n, p)) + np.sqrt(corr/ (1 - corr)) * normal(size=(n, 1))
  b[support] = np.ones(supp_size)
  mu = x.dot(b)
  var_xb = (np.std(mu, ddof=1))**2
  sd_error = np.sqrt(var_xb / snr)
  error = normal(size=n, scale=sd_error)
  y = mu + error
  return x, y, b


'''
Preprocess data in order to make sure all the variables in the same range
'''
def preprocess_data(x, y, intercept, normalize):
  
  if intercept:
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y)
    y_center = y - np.mean(y)
    x_center = x - np.mean(x, axis=0)
  else:
    y_mean = 0
    y_center = y
    x_mean = np.zeros(x.shape[1]) 
    x_center = x
  
  x_l2_norm = np.linalg.norm(x_center, axis=0)
  x_l2_norm_square = np.square(x_l2_norm)

  if normalize:
    y_l2_norm = np.linalg.norm(y_center)
    y_center = y_center / y_l2_norm
    x_center = x_center / x_l2_norm
    beta_multiplier = y_l2_norm / x_l2_norm
    x_l2_norm_square = np.ones(x.shape[1])
  else:
    beta_multiplier = np.ones(x.shape[1])

  return x_center, y_center, x_mean, y_mean, x_l2_norm_square, beta_multiplier

'''
Find the warm start of the current matrix model
'''
def supp_warm_start(x, y, x_l2_norm_square):
  # Compute the initial lambda_0 (typically leads to 1 nonzero).
  lambda_2=0.05
  temp_corrs = np.square(y.T.dot(x)) / (x_l2_norm_square + 2 * lambda_2)
  max_coef_index = np.argsort(temp_corrs)[-1:][::-1]
  curr_s = np.zeros(np.shape(x)[1])
  curr_s[max_coef_index] = 1 
  return curr_s