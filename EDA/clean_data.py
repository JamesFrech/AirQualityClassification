import pandas as pd

data = pd.read_csv('../data/pollution_dataset.csv')

# Replace bad values with the means in the columns.
# Only issues with PM10 and SO2
data.loc[data['PM10']==-.2,'PM10'] = data['PM10'].mean()
data.loc[data['SO2']<0,'SO2'] = data['SO2'].mean()

# Save the cleaned data.
data.to_csv('../data/clean.csv',index=False)
