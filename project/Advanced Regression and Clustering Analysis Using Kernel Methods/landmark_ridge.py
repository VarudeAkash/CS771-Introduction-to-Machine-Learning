import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def rbf_kernel(data1, data2):
    return np.exp(-0.1 * np.square(data1.reshape((-1, 1)) - data2.reshape(1, -1)))

def ridge_regression(x_train, y_train, x_test, lam, num_features):
    selected_indices = np.random.choice(len(x_train), num_features, replace=False)
    selected_x = x_train[selected_indices]
    rbf_features_train = rbf_kernel(x_train, selected_x)
    identity_matrix = np.identity(num_features)
    W = np.dot(np.linalg.inv(np.dot(rbf_features_train.T, rbf_features_train) + lam*identity_matrix), np.dot(rbf_features_train.T, y_train.reshape((-1,1))))
    rbf_features_test = rbf_kernel(x_test, selected_x)
    y_pred = np.dot(rbf_features_test, W)
    return y_pred

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def plot_results(x_test, y_test, y_pred, title):
    plt.title(title)
    plt.scatter(x_test, y_pred, s=10, c='red', label='Predictions')
    plt.scatter(x_test, y_test, s=10, c='blue', label='True Values')
    plt.legend()
    plt.show()

train = np.genfromtxt('data/ridgetrain.txt', delimiter='  ')
test = np.genfromtxt('data/ridgetest.txt', delimiter='  ')
print("data read successfully")
x_train = train[:, 0]
y_train = train[:, 1]
x_test = test[:, 0]
y_test = test[:, 1]

number_of_landmarks = [2, 5, 20, 50, 100]

print("Training Model.....Please wait. It may take few seconds")
for landmarks in number_of_landmarks:
    lam = 0.1
    y_pred = ridge_regression(x_train, y_train, x_test, lam, landmarks)
    rmse = calculate_rmse(y_test, y_pred)
    print(f'RMSE for number_of_landmarks {landmarks} = {rmse}')
    plot_results(x_test, y_test, y_pred, f'Number of landmarks = {landmarks}, RMSE = {rmse}')
