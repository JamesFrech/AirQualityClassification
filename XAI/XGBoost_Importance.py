import pandas as pd
import XGBoost as xgb
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
xgb = XGBClassifier(n_estimators=700,
                    min_child_weight=5,
                    max_depth=3,
                    learning_rate=0.03,
                    gamma=1.25,
                    objective='multi:softmax'
                    random_state=42)

xgb.fit(X_train, y_train.values.ravel())

# create dataframe of features
xgb_importance = pd.DataFrame(index=X_train.columns, data=xgb.feature_importances_, columns=["Importance"])
xgb_importance.sort_values(by="Importance", ascending=False, inplace=True)
print(xgb_importance)
print(xgb_importance.index)
print(xgb_importance.values)
# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(xgb_importance.index, xgb_importance.values[0]*100)
plt.xlabel('Feature Importance (%)')
plt.ylabel('Feature')
plt.title('XGBoost Feature Importance')
plt.savefig('images/XGBoost_FeatureImportance.png',bbox_inches='tight')
plt.close()

r = permutation_importance(xgb, X_test, y_test,
                           n_repeats=30,
                           random_state=0)

# test permutation importance
xgb_permutation_df = pd.DataFrame(index=X_train.columns,  columns=["Mean", "Standard Deviation"])
xgb_permutation_df["Mean"]=r["importances_mean"]
xgb_permutation_df["Standard Deviation"] = r["importances_std"]
xgb_permutation_df.sort_values(by="Mean", ascending=False, inplace=True)
xgb_permutation_df

# Run shap for random forest
explainer = shap.TreeExplainer(xgb)
explanation = explainer(X_test)
shap.plots.beeswarm(explanation, max_display=20)
plt.savefig('images/XGBoost_Shap_Beeswarm.png')
plt.close()
