import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
import matplotlib.pyplot as plt

# Select inputs
inputs = ['Temperature','Humidity','PM2.5','PM10','NO2','SO2',
          'CO','Proximity_to_Industrial_Areas','Population_Density']
target = ['Air Quality']
classes = ['Good','Moderate','Poor','Hazardous']

# Get the data
data = pd.read_csv('../data/clean_with_cluster_label.csv')
# Replace class label with class number for metric calculation
AQ_dict={'Good':0,'Moderate':1,'Poor':2,'Hazardous':3}
data['Air Quality'] = data['Air Quality'].replace(AQ_dict)

# Split the training and testing data.
X_train, X_test, y_train, y_test = train_test_split(
    data[inputs], data[target], test_size=0.2, random_state=42)

# Set param grid
# n_estimators for number of trees
# min_samples_leaf to avoid overfitting on small leaves and stop deep splits
# max_features to tune number of features considered at each split
param_grid = {'n_estimators':[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
              'min_samples_leaf':[2,3,4,5,10,15],
              'max_features':[i+1 for i in range(len(inputs))]}

# Initialize random forest classifier
rf = RandomForestClassifier()

# Initialize Kfold object with 5 folds
kfold = KFold(5,
              random_state=0,
              shuffle=True)

# Get the grid of hyperparameters for the model
# Need to specify scoring for f1 with multiclass
scoring = make_scorer(f1_score, average='weighted')
grid = RandomizedSearchCV(rf,
                          param_grid,
                          refit=True,
                          cv=kfold,
                          n_iter = 100,
                          verbose = 5,
                          scoring=scoring)

# Fit the model
grid.fit(X_train, y_train)

# Give best parameters and compare to others
print(grid.best_params_)
print(grid.cv_results_[('mean_test_score')])

# Fit to testing data
best_rf = grid.best_estimator_

# Predict classes
train_pred = best_rf.predict(X_train)
test_pred = best_rf.predict(X_test)

# Get F1, Precision, Recall
f1 = f1_score(y_train,train_pred,average=None)
prec = precision_score(y_train,train_pred,average=None)
rec = recall_score(y_train,train_pred,average=None)

train_metrics = pd.DataFrame(np.array([f1,prec,rec]).T,index=classes,columns=['F1','Precision','Recall'])


# Get F1, Precision, Recall
f1 = f1_score(y_test,test_pred,average=None)
prec = precision_score(y_test,test_pred,average=None)
rec = recall_score(y_test,test_pred,average=None)

test_metrics = pd.DataFrame(np.array([f1,prec,rec]).T,index=classes,columns=['F1','Precision','Recall'])

# Compare train and test metrics
print(train_metrics)
print(test_metrics)

print("Confusion Matrix\n",confusion_matrix(y_test, test_pred))
# Plot the confusion matrix
cm = confusion_matrix(y_test,test_pred)
ConfusionMatrixDisplay(cm,display_labels=classes).plot()
plt.savefig('images/RandomForest_confusion_matrix_test.png',bbox_inches='tight')
plt.close()
