# Stock Price Prediction
#### Video Demo: https://www.youtube.com/watch?v=A9_SUzB5MRc

This project is focused on predicting stock prices using a Support Vector Regression (SVR) model. The `stock.py` script contains the code implementation for this prediction task. Below is a brief overview of the script and its functionalities.

## Description:

### Dependencies

The following dependencies are required to run the script:
- numpy
- pandas
- scikit-learn (sklearn)
- matplotlib

You can install these dependencies using the package manager of your choice (e.g., pip).

### Dataset

The script generates a synthetic dataset of stock prices using random numbers. It creates 100 data points representing consecutive days and their corresponding stock prices. The prices are generated by adding the cumulative sum of random numbers to an initial price of 100.

### Model Training

The dataset is split into training and testing sets using a test size of 20%. The training set is then preprocessed using the MinMaxScaler to normalize the input features. The SVR model with a linear kernel is trained on the scaled training data.

### Model Evaluation

The trained model is used to make predictions on both the training and testing sets. The mean squared error (MSE) is calculated to evaluate the performance of the model on both sets. The MSE values are printed to the console.

### Prediction

The model is further used to predict stock prices for the next 10 consecutive days. New days are generated, and their corresponding prices are predicted using the trained model. The predicted prices are printed to the console.

### Visualization

Finally, a scatter plot is generated to visualize the training data, testing data, and the predicted prices. The training and testing data points are displayed as scattered points, while the predicted prices are represented by a line plot. The x-axis represents the days, and the y-axis represents the stock prices. The plot is titled "Stock Price Prediction" and includes a legend for better understanding.

Feel free to modify the code and experiment with different parameters to improve the prediction accuracy or explore alternative regression models.

**Note:** Keep in mind that this project uses a synthetic dataset for demonstration purposes. In real-world scenarios, obtaining accurate stock price predictions involves more complex factors and considerations.