import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR


np.random.seed(42)
num_data_points = 100
days = np.arange(1, num_data_points + 1)
prices = 100 + np.cumsum(np.random.rand(num_data_points))