import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Select model inputs
inputs = ['Temperature','Humidity','PM2.5','PM10','NO2','SO2',
          'CO','Proximity_to_Industrial_Areas','Population_Density']
target = ['Air Quality']

data = pd.read_csv('../data/clean.csv')

# Split the train/test data for EDA to avoid looking at the test set
X_train, X_test, y_train, y_test = train_test_split(
    data[inputs], data[target], test_size=0.2, random_state=42)

train = pd.concat([X_train,y_train],axis=1)

# Plot histogram for each input and separate colors for each air quality value
fig,axs=plt.subplots(3,3,figsize=(10,10))
k=0
for i in range(3):
    for j in range(3):
        for quality in ['Good','Moderate','Poor','Hazardous']:
            axs[i,j].hist(train.loc[train['Air Quality']==quality,inputs[k]],bins=20,label=quality)
        #axs[i,j].hist(data[inputs[k]],bins=20)
        axs[i,j].set_title(inputs[k])
        k+=1
plt.tight_layout()
plt.savefig('images/histograms.png',bbox_inches='tight')
plt.close()


# plot the correlation matrix
sns.heatmap(train[inputs].corr(),vmin=-1,vmax=1,cmap='bwr')
plt.savefig('images/correlation_matrix.png',bbox_inches='tight')
plt.close()

# plot the pair plots
sns.pairplot(train)
plt.savefig('images/pair_plots.png',bbox_inches='tight')
plt.close()
