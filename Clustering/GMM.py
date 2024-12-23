import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Use the variables that show separate normal distributions for each
# air quality category for use in GMM
inputs = ['Temperature','NO2','SO2','CO']
target = ['Air Quality']
classes = ['Good','Moderate','Poor','Hazardous']

# Get the data
data = pd.read_csv('../data/clean.csv')

# Split the training and testing data.
X_train, X_test, y_train, y_test = train_test_split(
    data[inputs], data[target], test_size=0.2, random_state=42)

# Initialize a GMM with 4 components, one for each air quality index.
gmm = GaussianMixture(n_components=4, n_init=10, random_state=42)

# Fit the model and predict training labels
gmm.fit(X_train[inputs])
train_predictions = gmm.predict(X_train[inputs])

# Print cluster means
print("Cluster means\n",gmm.means_)

# Get the index mapping from cluster number to air quality category number
# Use temperature (first column in means_) to order values as the order aligns
# with the order of category values
AQ_dict={'Good':0,'Moderate':1,'Poor':2,'Hazardous':3}
map_index = sorted(range(len(gmm.means_[0,:])), key=lambda k: gmm.means_[:,0][k],reverse=True)
map_dict={0:map_index[0],1:map_index[1],2:map_index[2],3:map_index[3]}

# Replace the cluster number with air quality category number
train_predictions = np.vectorize(map_dict.get)(train_predictions)

# Replot histograms with cluster means
# Plot histogram for each input and separate colors for each air quality value
train = pd.concat([X_train,y_train],axis=1) # For ease of plotting
fig,axs=plt.subplots(ncols=len(inputs),figsize=(10,3))
for i in range(len(inputs)):
    # Plot means from GMM model for each variable
    means = gmm.means_[:,i]
    means = [means[idx] for idx in map_index[::-1]]
    for k in range(4):
        axs[i].axvline(means[k],linestyle='--',c=f'C{k}')

    # Plot histogram for each category for each variable
    for quality in ['Good','Moderate','Poor','Hazardous']:
        axs[i].hist(train.loc[train['Air Quality']==quality,inputs[i]],bins=20,label=quality)
    axs[i].set_title(inputs[i])
plt.tight_layout()
plt.savefig('images/gmm_means_histograms.png',bbox_inches='tight')
plt.close()

# Replace class label with class number for metric calculation
y_train = y_train.replace(AQ_dict)

print("Confusion Matrix\n",confusion_matrix(y_train, train_predictions))
# Plot the confusion matrix
cm = confusion_matrix(y_train,train_predictions)
ConfusionMatrixDisplay(cm,display_labels=classes).plot()
plt.savefig('images/gmm_confusion_matrix.png',bbox_inches='tight')
plt.close()

# Get F1, Precision, Recall
f1 = f1_score(y_train,train_predictions,average=None)
prec = precision_score(y_train,train_predictions,average=None)
rec = recall_score(y_train,train_predictions,average=None)

train_metrics = pd.DataFrame(np.array([f1,prec,rec]).T,index=classes,columns=['F1','Precision','Recall'])

# Add the train cluster numbers to the data for classification
train=pd.concat([X_train,y_train],axis=1)
train['Cluster']=train_predictions



##################
# Test the model #
##################

# Get predictions
test_predictions = gmm.predict(X_test[inputs])

# Replace the cluster number with air quality category number
test_predictions = np.vectorize(map_dict.get)(test_predictions)

# Replace class label with class number for metric calculations
y_test = y_test.replace(AQ_dict)

print("Confusion Matrix\n",confusion_matrix(y_test, test_predictions))
# Plot the confusion matrix
cm = confusion_matrix(y_test,test_predictions)
ConfusionMatrixDisplay(cm,display_labels=classes).plot()
plt.savefig('images/gmm_confusion_matrix_test.png',bbox_inches='tight')
plt.close()

# Get F1, Precision, Recall
f1 = f1_score(y_test,test_predictions,average=None)
prec = precision_score(y_test,test_predictions,average=None)
rec = recall_score(y_test,test_predictions,average=None)

test_metrics = pd.DataFrame(np.array([f1,prec,rec]).T,index=classes,columns=['F1','Precision','Recall'])

# Compare train and test metrics
print(train_metrics)
print(test_metrics)

# Concat test data back together
test=pd.concat([X_test,y_test],axis=1)
test['Cluster']=test_predictions

# Concatenate all data back together
all_data = pd.concat([train,test])
all_data.sort_index(inplace=True)

# Convert category number back before merge
AQ_dict_to_category={0:'Good',1:'Moderate',2:'Poor',3:'Hazardous'}
all_data['Air Quality'] = all_data['Air Quality'].replace(AQ_dict_to_category)

# Add cluster labels to original dataset
data = data.merge(all_data)
# Output data for training other models
data.to_csv('../data/clean_with_cluster_label.csv',index=False)
