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
rf_importance.sort_values(by="Importance", ascending=True, inplace=True)
print(rf_importance)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(rf_importance.index, rf_importance['Importance']*100)
plt.xlabel('Feature Importance (%)')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importance')
plt.savefig('images/RandomForest_FeatureImportance.png',bbox_inches='tight')
plt.close()

# Compute permutation importance
r = permutation_importance(rf, X_test, y_test,
                           n_repeats=30,
                           random_state=0)

# test permutation importance
rf_permutation = pd.DataFrame(index=X_train.columns,  columns=["Mean", "Standard Deviation"])
rf_permutation["Mean"]=r["importances_mean"]
rf_permutation["Standard Deviation"] = r["importances_std"]
rf_permutation.sort_values(by="Mean", ascending=True, inplace=True)
print(rf_permutation)

# Plot Permutation importances
plt.figure(figsize=(10, 6))
plt.axvline(0,c='black')
plt.barh(rf_permutation.index, rf_permutation['Mean']*100)
plt.xlabel('Permutation Importance (%)')
plt.ylabel('Feature')
plt.title('Random Forest Permutation Importance')
plt.savefig('images/RandomForest_PermutationImportance.png',bbox_inches='tight')
plt.close()

########
# SHAP #
########

# Run shap for random forest
explainer = shap.TreeExplainer(rf)
explanation = explainer(X_test)

# Plot good shap values
shap.plots.beeswarm(explanation[:,:,0], max_display=20, show=False)
plt.title('Random Forest Shap for Good Air Quality')
plt.savefig('images/RandomForest_Shap_Beeswarm_GoodAQ.png',bbox_inches='tight')
plt.close()

# Plot moderate shap values
shap.plots.beeswarm(explanation[:,:,1], max_display=20, show=False)
plt.title('Random Forest Shap for Moderate Air Quality')
plt.savefig('images/RandomForest_Shap_Beeswarm_ModerateAQ.png',bbox_inches='tight')
plt.close()

# Plot poor shap values
shap.plots.beeswarm(explanation[:,:,2], max_display=20, show=False)
plt.title('Random Forest Shap for Poor Air Quality')
plt.savefig('images/RandomForest_Shap_Beeswarm_PoorAQ.png',bbox_inches='tight')
plt.close()

# Plot hazardous shap values
shap.plots.beeswarm(explanation[:,:,3], max_display=20, show=False)
plt.title('Random Forest Shap for Hazardous Air Quality')
plt.savefig('images/RandomForest_Shap_Beeswarm_HazardousAQ.png',bbox_inches='tight')
plt.close()
