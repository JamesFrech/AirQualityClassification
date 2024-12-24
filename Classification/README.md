# Classification

This directory considers many machine learning models for air quality classification. Logistic regression performs rather well, but struggles with poor and hazardous categories. SVM performs better, however, random forest and XGBoost both perform very well (and almost exactly the same).

---

## Files

### LogisticRegression.py
Fits a logistic regression model to the air quality data. Population density was excluded as an input in the final model since its p-value did not pass the significance threshold for any of the categories of air quality.
![alt text](images/LogisticRegression_confusion_matrix_test.png "Logistic Regression Confusion Matrix")

### SVM_linear.py
Fits a linear SVM to the air quality data using a 5 fold cross validation to tune the cost parameter "C".
![alt text](images/SVM_Linear_confusion_matrix_test.png "Linear SVM Confusion Matrix")

### RandomForest.py
Fits a random forest classifier to the air quality data using a 5 fold cross validation tuning the number of trees, minimum samples per leaf, and number of inputs allowed at each input. A randomized grid search is used to increase computational efficiency.
![alt text](images/RandomForest_confusion_matrix_test.png "Random Forest Confusion Matrix")

### XGBoost.py
Fits an XGBoost classifier to the air quality data with 5 fold cross validation used to tune the number of trees, max depth, learning rate, minimum child weight, and gamma hyperparameters. A randomized grid search is used.
![alt text](images/XGBoost_confusion_matrix_test.png "XGBoost Confusion Matrix")

### NeuralNetworkModel.py
Contains classes for a pytorch dataset and a class for the basic neural network used for classification in NeuralNetwork.py

### NeuralNetwork.py
Trains a neural network for multi-class classification of air quality data using pytorch. The model contains one hidden layer with 5 nodes and shows comparable performance to the other models achieving a 95% test accuracy in 300 epochs. A learning rate decay is applied at 200 epochs to prevent high variablity in test accuracy and to fine tune parameters.
![alt text](images/NN_Confusion_Matrix_test.png "Neural Network Confusion Matrix")

Loss and Accuracy by epoch
![alt text](images/NN_accuracy.png "Neural Network Accuracy")
![alt text](images/NN_loss.png "Neural Network Loss")
