# Advanced Regression and Clustering Analysis Using Kernel Methods

## Project Overview

This project focuses on implementing advanced regression and clustering techniques using kernel methods. It demonstrates how kernel functions can transform data into higher-dimensional spaces, enabling more powerful and flexible algorithms for regression and clustering tasks. The methods applied include kernel ridge regression, landmark-based ridge regression, kernel K-means clustering, and handcrafted feature-based K-means clustering. The dataset used for this project contains data points for regression and clustering analysis.

## Methods and Techniques

- **Kernel Ridge Regression**: This method leverages the kernel trick to apply ridge regression in higher-dimensional feature spaces, providing better flexibility in fitting nonlinear data.
- **Landmark-Based Ridge Regression**: Using random data points as landmarks, this method computes kernel-based features for ridge regression, balancing accuracy and computational efficiency.
- **Kernel K-Means Clustering**: This variant of K-means clustering leverages kernel functions to separate data in complex, non-linearly separable feature spaces.
- **Handcrafted Feature K-Means**: Transforms data into handcrafted feature spaces to facilitate clustering, providing insight into feature engineering's impact on clustering performance.

## Implementation

1. **Kernel Ridge Regression**: Applied RBF kernel to the training data and optimized using ridge regression for different lambda values. The performance is evaluated using Root Mean Square Error (RMSE) on the test set.
2. **Landmark-Based Ridge Regression**: Selected random data points as landmarks to compute the RBF kernel matrix and applied ridge regression with varying landmark sizes. RMSE values were used for performance evaluation.
3. **Kernel K-Means Clustering**: Randomly selected landmark features were used to compute the kernel matrix for clustering, optimizing cluster assignments iteratively until convergence.
4. **Handcrafted Feature K-Means Clustering**: Squared the input feature space, applying the K-means algorithm to group data points into clusters and visualize the results in both original and transformed feature spaces.

## Usage Instructions

1. **Kernel Ridge Regression**:
   - Run the `Kernel_ridge.py` file to perform kernel ridge regression on the dataset.
   - Modify the lambda values and observe the impact on RMSE and prediction performance.

2. **Landmark-Based Ridge Regression**:
   - Run the `Landmark_ridge.py` file to apply ridge regression using kernel features derived from a set of randomly selected landmarks.
   - Vary the number of landmarks to explore the trade-offs between accuracy and computational complexity.

3. **Kernel K-Means Clustering**:
   - Execute the `Kernel_Kmeans.py` file to perform clustering using the kernelized version of the K-means algorithm.
   - Analyze the clustering results using visualizations and compare the effect of different landmark selections.

4. **Handcrafted Feature K-Means**:
   - Execute the `Handcrafted_Means.py` file to apply K-means clustering on transformed data, created by applying feature engineering techniques.
   - Visualize the clustering results in both the original and transformed spaces.

## Prerequisites

- Python 3.x
- NumPy
- Matplotlib
- Pandas
- Scikit-learn

## Dataset

The dataset for this project is included in the `data/` folder, containing:
- `ridgetrain.txt`: Training data for regression tasks.
- `ridgetest.txt`: Testing data for regression tasks.
- `kmeans_data.txt`: Data points for clustering analysis.

## Results

This project demonstrated the effectiveness of kernel methods in both regression and clustering tasks. The kernel ridge regression models achieved reduced RMSE across different lambda values and landmark sizes. The kernel K-means clustering effectively separated non-linear data, showing superior performance compared to basic K-means in high-dimensional feature spaces.

## Acknowledgments

This project was part of an advanced machine learning course CS771, emphasizing the application of kernel methods in various machine learning tasks. Special thanks to the course instructors for their guidance and the provided datasets.

**Note**: If you use any part of the code from my project, please provide appropriate acknowledgment and attribution.

