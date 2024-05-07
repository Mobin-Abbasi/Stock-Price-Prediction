import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import matplotlib.pyplot as plt


def main():
    num_data_points = 100
    days = np.arange(1, num_data_points + 1)
    r = np.random.randn(num_data_points)
    prices = 100 + np.cumsum(r)

    data = pd.DataFrame({'Days': days, 'Prices': prices})
    data['Days'] = data['Days'].astype(float)

    model, scaler, X_train, y_train, X_test, y_test, y_pred_train, y_pred_test, _, _ = train_model(data)

    new_days = np.arange(num_data_points + 1, num_data_points + 11)
    new_prices = predict_prices(model, scaler, new_days)

    display_results(X_train, y_train, X_test, y_test, y_pred_train, y_pred_test, new_days, new_prices)


def train_model(data):
    X = data[['Days']]
    y = data['Prices']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVR(kernel='linear')
    model.fit(X_train_scaled, y_train)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    return model, scaler, X_train, y_train, X_test, y_test, y_pred_train, y_pred_test, mse_train, mse_test


def predict_prices(model, scaler, new_days):
    new_days_df = pd.DataFrame({'Days': new_days})
    new_days_scaled = scaler.transform(new_days_df[['Days']])
    new_prices = model.predict(new_days_scaled)
    return new_prices


def display_results(X_train, y_train, X_test, y_test, y_pred_train, y_pred_test, new_days, new_prices):
    print("Training MSE:", mean_squared_error(y_train, y_pred_train))
    print("Testing MSE:", mean_squared_error(y_test, y_pred_test))
    print("Predicted Prices:", new_prices)

    plt.figure(figsize=(15, 10))
    plt.scatter(X_train, y_train, color='blue', label='Training Data')
    plt.scatter(X_test, y_test, color='green', label='Testing Data')
    plt.scatter(new_days, new_prices, color='pink', label='Predicting Data')
    plt.plot(X_train, y_pred_train, color='red', label='Training Prediction')
    plt.plot(X_test, y_pred_test, color='orange', label='Testing Prediction')
    plt.xlabel('Days')
    plt.ylabel('Prices')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
