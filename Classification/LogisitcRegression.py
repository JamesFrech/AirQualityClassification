import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, precision_score, recall_score
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Select inputs
# Population density showed a p-value not significant for each class,
# so remove
inputs = ['Temperature','Humidity','PM2.5','PM10','NO2','SO2',
          'CO','Proximity_to_Industrial_Areas']#,'Population_Density']
target = ['Air Quality']
classes = ['Good','Moderate','Poor','Hazardous']

# Get the data
data = pd.read_csv('../data/clean_with_cluster_label.csv')
# Add a constant for logistic regression
data = sm.add_constant(data)
# Replace class label with class number for metric calculation
AQ_dict={'Good':0,'Moderate':1,'Poor':2,'Hazardous':3}
data['Air Quality'] = data['Air Quality'].replace(AQ_dict)

# Split the training and testing data.
X_train, X_test, y_train, y_test = train_test_split(
    data[inputs], data[target], test_size=0.2, random_state=42)

# Fit logistic regression
logit_stats = sm.MNLogit(y_train, X_train)
log_r = logit_stats.fit()

print(log_r.summary())

# Get probabilities
train_pred = log_r.predict(X_train)
test_pred = log_r.predict(X_test)
# Get max of probabilities
train_pred['prediction']=train_pred.idxmax(axis=1)
test_pred['prediction']=test_pred.idxmax(axis=1)

# Get F1, Precision, Recall
f1 = f1_score(y_train,train_pred['prediction'],average=None)
prec = precision_score(y_train,train_pred['prediction'],average=None)
rec = recall_score(y_train,train_pred['prediction'],average=None)

train_metrics = pd.DataFrame(np.array([f1,prec,rec]).T,index=classes,columns=['F1','Precision','Recall'])


# Get F1, Precision, Recall
f1 = f1_score(y_test,test_pred['prediction'],average=None)
prec = precision_score(y_test,test_pred['prediction'],average=None)
rec = recall_score(y_test,test_pred['prediction'],average=None)

test_metrics = pd.DataFrame(np.array([f1,prec,rec]).T,index=classes,columns=['F1','Precision','Recall'])

# Compare train and test metrics
print(train_metrics)
print(test_metrics)

print("Confusion Matrix\n",confusion_matrix(y_test, test_pred['prediction']))
# Plot the confusion matrix
cm = confusion_matrix(y_test,test_pred['prediction'])
ConfusionMatrixDisplay(cm,display_labels=classes).plot()
plt.savefig('images/LogisticRegression_confusion_matrix_test.png',bbox_inches='tight')
plt.close()
