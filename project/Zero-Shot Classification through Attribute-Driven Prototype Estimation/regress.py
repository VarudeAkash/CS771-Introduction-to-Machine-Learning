#Name: Akash Shivaji Varude
#Programme: Mtech CSE
#Roll Number: 231110006

import numpy as np
from sklearn.metrics import accuracy_score

#added encoding as latin1 cause without adding it was showing some error
X_seen=np.load('X_seen.npy',allow_pickle=True,encoding="latin1")#(40 x N_i x D): 40 feature matrices. X_seen[i] is the N_i x D feature matrix of seen class i
Xtest=np.load('Xtest.npy')   #(6180, 4096): feature matrix of the test data.
Ytest=np.load('Ytest.npy',)   #(6180, 1): ground truth labels of the test data
class_attributes_seen=np.load('class_attributes_seen.npy')  #(40, 85): 40x85 matrix with each row being the 85-dimensional class attribute vector of a seen class.
class_attributes_unseen=np.load('class_attributes_unseen.npy')   #(10, 85): 10x85 matrix with each row being the 85-dimensional class attribute vector of an  unseen class.

print("\nData is loaded")
#to calculate mean vectors of seen classes, these vectors are stored in the matrix form where each row i of  matrix seen_means
#corresponds to mean vector of ith class 
seen_means= np.zeros(shape=(40, 4096))
for i in range(40):
    class_mean=np.mean(X_seen[i],axis=0)
    seen_means[i]=class_mean
print("Calculated seen classes means")
#just creating variable names as mentioned in question
A_s= class_attributes_seen     
M_s=seen_means

#print(A_s.shape, M_s.shape)
print("Training the model")
lambdas=[0.01, 0.1, 1, 10, 20, 50, 100]
I=np.identity(85)
temp=np.matmul(A_s.T,M_s)      #doing A_s * M_s matrix multiplication just to avoid overhead ahead
accuracies=[]
for lamb in lambdas:
    W=np.matmul(np.linalg.inv(np.matmul(A_s.T,A_s)+lamb*I), temp)    #using formula given in the question itself
    #print(W.shape)
    W=W.T
    unseen_means = np.matmul(class_attributes_unseen, W.T)          #calculating means of unseen classes
    #Here instead of calculating dist manually one by one, I am using numpy vectorization concept for speeding up calculation 
    distances = np.linalg.norm(Xtest[:, np.newaxis] - unseen_means, axis=2)   #calculating distance of test input to all unseen class means
    y_predict = np.argmin(distances, axis=1) + 1                    # finding index of class with minimum distance. I'm adding 1 to it as index start from 0
    accuracies.append(accuracy_score( y_predict, Ytest)*100)
    print("For lambda ",lamb, " Test accuracy = ",accuracy_score( y_predict, Ytest)*100,"%")
print("all accuracies :",accuracies)