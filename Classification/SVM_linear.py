import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
import matplotlib.pyplot as plt

# Select inputs
# Population density showed a p-value not significant for each class,
# so remove
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

# Scale the data, only fit using training data to avoid leakage
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Initialize a linear SVM
svm = SVC(kernel='linear')

# Initialize Kfold object with 5 folds
kfold = KFold(5,
              random_state=0,
              shuffle=True)

# Get the grid of hyperparameters for the model
# Need to specify scoring for f1 with multiclass
#scoring = {'f1_score' : make_scorer(f1_score, average='weighted')}
scoring = make_scorer(f1_score, average='weighted')
grid = GridSearchCV(svm,
                    {'C':[0.001,0.01,0.1,1,5,10,100]},
                    refit=True,
                    cv=kfold,
                    scoring=scoring)

# Fit the model
grid.fit(X_train, y_train.values.ravel())

# Give best parameters and compare to others
print(grid.best_params_)
print(grid.cv_results_[('mean_test_score')])

# Fit to testing data
best_svc = grid.best_estimator_

# Predict classes
train_pred = best_svc.predict(X_train)
test_pred = best_svc.predict(X_test)

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
plt.savefig('images/SVM_Linear_confusion_matrix_test.png',bbox_inches='tight')
plt.close()
