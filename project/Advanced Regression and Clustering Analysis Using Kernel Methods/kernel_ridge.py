import numpy as np
import matplotlib.pyplot as plt

train = np.genfromtxt('data/ridgetrain.txt', delimiter='  ')
test = np.genfromtxt('data/ridgetest.txt', delimiter='  ')
print("Data imported successfylly")
x_train=train[:, 0]
y_train=train[:, 1]
x_test=test[:, 0]
y_test = test[:, 1]
lambda_val = [0.1, 1, 10, 100]

def rbf_kernel(data1, data2):
    return np.exp(-0.1 * np.square(data1.reshape((-1, 1)) - data2.reshape(1, -1)))

def ridge_regression(x_train, y_train, x_test, lam):
    training_kernel_matrix = rbf_kernel(x_train, x_train)        #jist calculated K matrix such that K_ij= k(xi, xj )
    identity_matrix = np.identity(x_train.shape[0])     #created Identity matrix of size NxN where N is number of training examples
    alpha = np.dot(np.linalg.inv(training_kernel_matrix + lam * identity_matrix), y_train.reshape((-1, 1)))  #calculated alpha
    Testing_kernel_matrix = rbf_kernel(x_train, x_test)    #jist calculated K_test matrix suck that K_ij= k(xi, xj )
    y_pred = np.dot(alpha.T, Testing_kernel_matrix).reshape((-1, 1))
    return y_pred


for l in lambda_val:
    print("Training Model....Please wait for few seconds\n")
    y_pred = ridge_regression(x_train, y_train, x_test, l)
    RMSE = np.sqrt(np.mean(np.square(y_test.reshape((-1, 1)) - y_pred)))
    print('for lambda = ' + str(l) + ', RMS error = ' + str(RMSE))
    plt.title('lambda = ' + str(l) + ', RootMeanSquareError = ' + str(RMSE))
    plt.scatter(x_test, y_pred,s=10, c='red')
    plt.scatter(x_test, y_test,s=10,c= 'blue')
    plt.show()
plt.show()
