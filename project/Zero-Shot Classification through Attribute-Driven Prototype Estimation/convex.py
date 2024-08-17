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

#print(X_seen.shape,Xtest.shape,Ytest.shape,class_attributes_seen.shape,class_attributes_unseen.shape)
print("\nData is loaded")
#to calculate mean vectors of seen classes, these vectors are stored in the matrix form where each row i of  matrix seen_means
#corresponds to mean vector of ith class 
seen_means= np.zeros(shape=(40, 4096))
for i in range(40):
    class_mean=np.mean(X_seen[i],axis=0)
    seen_means[i]=class_mean

print("Calculated seen classes means")
#calculate S(c,k)
unseen_s= np.zeros(shape=(10, 40))
for k in range(10):
    for i in range(40):
        unseen_s[k][i]=np.inner(class_attributes_unseen[k], class_attributes_seen[i])
    unseen_s[k]=unseen_s[k]/np.sum(unseen_s[k])     #just normalizing as per mentioned in the question


#calculating means for unseen data using convex combinations of seen means and similarity weights
unseen_means=np.zeros(shape=(10,4096))
unseen_means = np.dot(unseen_s, seen_means)  # Shape (10, 4096)
print("Calculated unseen classes means")
#Code in comments below is just for cross checking 
# unseen_means=np.zeros(shape=(10,4096))
# for c in range(10):
#     x=np.zeros(4096)
#     for k in range(40):
#         x=x+unseen_s[c][k]*seen_means[k]
#     unseen_means[c]=x

#Implementing prototype based model just on unseen means
y_predict=np.zeros(6180)
print("Training the model")
for i in range(6180):
    distances=[]
    for j in range(10):
        dist=np.linalg.norm(Xtest[i]-unseen_means[j])   #calculating distance of test input with each of unseen mean
        distances.append(dist)                          #storing theose in list
    predict=1+distances.index(min(distances))           #predicting the class of test input with minimum distance. added 1 to index as indexing starts from 0 
    y_predict[i]=predict

print("accuracy using method 1 is",accuracy_score( y_predict, Ytest)*100,"%")
# End measuring time
