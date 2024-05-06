import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import pytest
from stock import train_model, predict_prices, display_results


def test_train_model():
    num_data_points = 100
    days = np.arange(1, num_data_points + 1)
    r = np.random.randn(num_data_points)
    prices = 100 + np.cumsum(r)

    data = pd.DataFrame({'Days': days, 'Prices': prices})
    data['Days'] = data['Days'].astype(float)

    model, *_ = train_model(data)

    assert isinstance(model, SVR)


def test_predict_prices():
    num_data_points = 100
    days = np.arange(1, num_data_points + 1)
    r = np.random.randn(num_data_points)
    prices = 100 + np.cumsum(r)

    data = pd.DataFrame({'Days': days, 'Prices': prices})
    data['Days'] = data['Days'].astype(float)

    model, scaler, *_ = train_model(data)

    new_days = np.arange(num_data_points + 1, num_data_points + 11)
    new_prices = predict_prices(model, scaler, new_days)

    assert isinstance(new_prices, np.ndarray)
    assert len(new_prices) == 10


def test_display_results():
    num_data_points = 100
    days = np.arange(1, num_data_points + 1)
    r = np.random.randn(num_data_points)
    prices = 100 + np.cumsum(r)

    data = pd.DataFrame({'Days': days, 'Prices': prices})
    data['Days'] = data['Days'].astype(float)

    model, scaler, X_train, y_train, X_test, y_test, y_pred_train, y_pred_test, *_ = train_model(data)

    new_days = np.arange(num_data_points + 1, num_data_points + 11)
    new_prices = predict_prices(model, scaler, new_days)

    # Testing types of outputs (no assertion on plot)
    assert isinstance(display_results(X_train, y_train, X_test, y_test, y_pred_train, y_pred_test, new_days, new_prices), None.__class__)