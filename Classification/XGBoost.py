import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
import xgboost as xgb
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

# Initialize random forest classifier
xgb_forest = xgb.XGBClassifier(importance_type="gain")

# Initialize Kfold object with 5 folds
kfold = KFold(5,
              random_state=0,
              shuffle=True)

# Get the grid of hyperparameters for the model
# Need to specify scoring for f1 with multiclass
#scoring = {'f1_score' : make_scorer(f1_score, average='weighted')}
scoring = make_scorer(f1_score, average='weighted')

n_estimators = [400, 500, 600, 700, 900, 1000, 1100,]
max_depth = [2, 3, 4, 5, 10, 15]
learning_rate=[0.01, 0.02, 0.03, 0.05,0.1]
min_child_weight=[1,2,3,4,5,6]
gamma=[0.5, 0.75, 1, 1.25, 1.5]

hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'objective':["multi:softmax"],
    'gamma':gamma,
    }

# Initialize random search CV
random_cv = RandomizedSearchCV(estimator=xgb_forest,
            param_distributions=hyperparameter_grid,
            cv=5,
            n_iter=100,
            scoring = scoring,
            verbose = 5,
            return_train_score = True,
            random_state=42)

# Run the random search CV
random_cv.fit(X_train,y_train)

# Give best parameters and compare to others
print(random_cv.best_params_)
print(random_cv.cv_results_[('mean_test_score')])

# Fit to testing data
best_xgb = random_cv.best_estimator_

# Predict classes
train_pred = best_xgb.predict(X_train)
test_pred = best_xgb.predict(X_test)

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
plt.savefig('images/XGBoost_confusion_matrix_test.png',bbox_inches='tight')
plt.close()
