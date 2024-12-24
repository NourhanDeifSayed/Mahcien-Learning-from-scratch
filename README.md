# Mahcien-Learning-from-scratch

K-Nearest Neighbors (KNN) Classifier Project: Iris Dataset

Implemented a K-Nearest Neighbors (KNN) classifier from scratch to predict the species of Iris flowers based on various features. This project demonstrates the application of key machine learning concepts, such as distance metrics, classification, and model evaluation. The model was built using the Iris dataset and was compared against a pre-built KNN classifier from scikit-learn for accuracy validation.

Key Steps:
Data Preprocessing: Loaded the Iris dataset and split it into training and test sets using train_test_split.
Euclidean Distance Calculation: Defined a function to compute the Euclidean distance between data points, which is central to KNN.
KNN Algorithm: Built the KNN algorithm by implementing functions to find the nearest neighbors and predict labels based on majority voting.
Model Evaluation: Evaluated the model's performance using accuracy metrics and compared the custom KNN implementation with scikit-learn’s KNeighborsClassifier.

Skills Used:
Algorithms: K-Nearest Neighbors, Euclidean Distance, Voting System
Libraries: NumPy, pandas, scikit-learn
Model Evaluation: Accuracy, Model Validation
Data Preprocessing: Data Splitting, Feature Normalization
--------------------------------------------------------------------------------------------------------------------------------------------------------


Linear Regression and Least Squares Classifier for Iris Dataset:
This project demonstrates the implementation of linear regression models and a custom least squares classifier for predicting Iris species and analyzing linear relationships.

Linear Regression Model:
Linear Regression without Noise: Defined a simple linear equation y = 2x + 1 and trained a model using LinearRegression from sklearn.
Linear Regression with Noise: Introduced random noise to the data and retrained the model to show how noise affects the regression line and its parameters.
Visualization: Plotted both the original data and noisy data along with their respective regression lines to visually compare the effect of noise.

Least Squares Classifier:
Implemented a custom Least Squares Classifier to perform binary classification of Iris species using the Iris dataset.
Split the dataset into training and test sets and visualized the decision boundary for classifying two species of Iris.
Evaluated the classifier's performance with a scatter plot displaying the classified points for test data.

Libraries Used:
NumPy: For mathematical and array operations.
Matplotlib: For visualizing regression lines and decision boundaries.
Scikit-learn: For model training and data preprocessing.
Seaborn: For visualizing data relationships and patterns.

---------------------------------------------------------------------------------------------------------------------

Support Vector Machine (SVM) Classification on Multiple Datasets

This project demonstrates the use of Support Vector Machines (SVM) for classification on several datasets. The datasets used include Aggregation, Jain, Flame, Pathbased, and Compound, and are processed to predict binary classifications using the SVM with a radial basis function (RBF) kernel.

Key Steps:
Data Loading & Preprocessing:

Datasets are loaded using pandas and preprocessed by splitting into features (X) and labels (y).
Data is split into training and testing sets using train_test_split.
Model Training:

An SVM model with an RBF kernel is used for training on each dataset.
The model is trained on the training data and evaluated on the test data for performance.
Visualization:

The decision boundary is visualized using matplotlib, displaying how the classifier separates the classes for each dataset.
Random points are generated across the feature space, and their classifications are visualized alongside the actual training data.
Performance Evaluation:

The accuracy of the model is calculated by comparing predicted values (y_pred) to the actual test labels (y_test).
This code provides a comprehensive view of SVM classification on multiple datasets, showcasing the importance of kernel choice and data visualization in understanding classifier behavior.



_______________________________________________________________________________________________________________________________________________________________________

MNIST Handwritten Digit Classification Using Neural Networks


This project implements a custom neural network from scratch to classify handwritten digits from the MNIST dataset. It involves data preprocessing, neural network initialization, feedforward propagation, and backpropagation for weight updates. Model performance is evaluated using a Confusion Matrix and Accuracy Score.

Key Features:

Dataset Used: MNIST handwritten digit dataset.
Data Preprocessing:
Reshaped and normalized input features to prepare for training.
Split data into training and testing sets.
Neural Network Architecture:
Input layer size: 784 (28x28 images).
Hidden layer size: 120 neurons.
Output layer size: 10 neurons (digits 0–9).
Activation Function: Sigmoid for hidden and output layers.
Evaluation Metrics: Confusion Matrix and Accuracy Score.

Technologies Used:
Python
NumPy
Pandas
Matplotlib
scikit-learn
Keras (MNIST dataset loader)
Results:

Model predictions are evaluated on the test set with a confusion matrix.
Accuracy is computed to assess performance.
________________________________________________________________________________________________________________

