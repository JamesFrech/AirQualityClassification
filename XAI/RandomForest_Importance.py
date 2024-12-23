import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import shap

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

# Initialize random forest classifier with best parameters
rf = RandomForestClassifier(n_estimators=300,
                            min_samples_leaf=4,
                            max_features=2,
                            random_state=42)

rf.fit(X_train, y_train.values.ravel())

# create dataframe of features
rf_importance = pd.DataFrame(index=X_train.columns, data=rf.feature_importances_, columns=["Importance"])
rf_importance.sort_values(by="Importance", ascending=False, inplace=True)
print(rf_importance)
print(rf_importance.index)
print(rf_importance.values)
# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(rf_importance.index, rf_importance.values[0]*100)
plt.xlabel('Feature Importance (%)')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importance')
plt.savefig('images/RandomForest_FeatureImportance.png',bbox_inches='tight')
plt.close()

r = permutation_importance(rf, X_test, y_test,
                           n_repeats=30,
                           random_state=0)

# test permutation importance
forest_permutation_df = pd.DataFrame(index=X_train.columns,  columns=["Mean", "Standard Deviation"])
forest_permutation_df["Mean"]=r["importances_mean"]
forest_permutation_df["Standard Deviation"] = r["importances_std"]
forest_permutation_df.sort_values(by="Mean", ascending=False, inplace=True)
forest_permutation_df

# Run shap for random forest
explainer = shap.TreeExplainer(rf)
explanation = explainer(X_test)
shap.plots.beeswarm(explanation, max_display=20)
plt.savefig('images/RandomForest_Shap_Beeswarm.png')
plt.close()
