import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from NeuralNetworkModel import TorchDataset, NeuralNetwork
import matplotlib.pyplot as plt

import torch
torch.manual_seed(42)

# Select inputs
inputs = ['Temperature','Humidity','PM2.5','PM10','NO2','SO2',
          'CO','Proximity_to_Industrial_Areas']#,'Population_Density']
target = ['Air Quality']
classes = ['Good','Moderate','Poor','Hazardous']

# Get the data
data = pd.read_csv('../data/clean_with_cluster_label.csv')
# Replace class label with class number for metric calculation
AQ_dict={'Good':0.0,'Moderate':1.0,'Poor':2.0,'Hazardous':3.0}
data['Air Quality'] = data['Air Quality'].replace(AQ_dict)

# Split the training and testing data.
X_train, X_test, y_train, y_test = train_test_split(
    data[inputs], data[target], test_size=0.2, random_state=42)

# convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values.squeeze(), dtype=torch.float32).type(torch.LongTensor)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values.squeeze(), dtype=torch.float32).type(torch.LongTensor)

model = NeuralNetwork(n_inputs=len(inputs),n_outputs=len(classes))

train_data = TorchDataset(X_train, y_train)
test_data = TorchDataset(X_test, y_test)

trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
testloader = DataLoader(test_data, batch_size=64, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=200, gamma=0.1)

# Training the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize loss and accuracies
loss_epoch=[]
train_acc_epoch=[]
test_acc_epoch=[]

n_epoch=300
for epoch in range(n_epoch):  # Loop over the dataset multiple times
    print(f'epoch {epoch}')
    running_loss = 0.0
    batches = 0.0
    correct = 0.0
    total = 0.0
    correct_test = 0
    total_test = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        #print(inputs,labels)

        optimizer.zero_grad()

        outputs = model(inputs)
        #print(outputs,labels)

        loss = criterion(outputs, labels) # Compute loss
        #print(loss.item())
        #exit()
        loss.backward() # Backpropagation
        optimizer.step() # Update weights

        # Calculate the loss
        running_loss += loss.item()
        batches += 1
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        #print(predicted,labels)

        correct += (predicted == labels).sum().item()

    # Testing the model
    with torch.no_grad(): # Don't update weights when testing
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    # Print loss for given epoch
    print('loss: %.3f' %
          (running_loss / batches))
    print(f'Train accuracy: {100 * correct / total}')
    print(f'Test accuracy: {100 * correct_test / total_test}')

    loss_epoch.append(running_loss / batches)
    train_acc_epoch.append(100 * correct / total)
    test_acc_epoch.append(100 * correct_test / total_test)

    # Delete loss, outputs to make sure RAM stays stable.
    del running_loss, correct, correct_test, outputs

    scheduler.step()
    # Stop early if test accuracy is less than the last two epochs
    #if epoch >= 2:
    #  if test_acc_epoch[epoch] < test_acc_epoch[epoch-2] and test_acc_epoch[epoch] < test_acc_epoch[epoch-1]:
#        break

print('Finished Training')

# Loss
plt.plot([i for i in range(len(loss_epoch))],loss_epoch)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('images/NN_loss.png',bbox_inches='tight')
plt.close()

# Accuracy
plt.plot([i for i in range(len(train_acc_epoch))],train_acc_epoch,label='train')
plt.plot([i for i in range(len(test_acc_epoch))],test_acc_epoch,label='test')
plt.legend()
plt.title('Train and Test Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig('images/NN_accuracy.png')
plt.close()

# Confusion matrix
all_labels_train=[]
train_pred=[]
with torch.no_grad(): # Don't update weights when testing
    for data in trainloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels_train = all_labels_train + labels.tolist()
        train_pred = train_pred + predicted.tolist()

# Confusion matrix
all_labels=[]
pred=[]
with torch.no_grad(): # Don't update weights when testing
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels = all_labels + labels.tolist()
        pred = pred + predicted.tolist()

cm = confusion_matrix(all_labels, pred)
ConfusionMatrixDisplay(cm).plot()
plt.savefig('images/NN_Confusion_Matrix_test',bbox_inches='tight')
plt.close()

f1 = f1_score(all_labels_train,train_pred,average=None)
prec = precision_score(all_labels_train,train_pred,average=None)
rec = recall_score(all_labels_train,train_pred,average=None)

train_metrics = pd.DataFrame(np.array([f1,prec,rec]).T,index=classes,columns=['F1','Precision','Recall'])
print(train_metrics)

f1 = f1_score(all_labels,pred,average=None)
prec = precision_score(all_labels,pred,average=None)
rec = recall_score(all_labels,pred,average=None)

metrics = pd.DataFrame(np.array([f1,prec,rec]).T,index=classes,columns=['F1','Precision','Recall'])
print(metrics)
