# Air Quality Classification

This project implements many machine learning techniques on the kaggle dataset "Air quality and Pollution Assessment" (https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment) to classify air pollution as "Good", "Moderate", "Poor", or "Hazardous". Nine provided input features are considered for the models to predict the output "Air Quality". First, an exploratory data analysis was performed to analyze the distributions of input variables and their relationships with the output data. From there, clustering was performed to separate the data into different clusters that could represent the classes. After, the many models including logistic regression, linear SVM, random forest, XGBoost, and a neural network were used to classify the data. Precision, Recall, and F1 scores for each class are analyzed on a test dataset for each model in the scripts, however confusion matrices are provided instead in the READMEs.

---

## data
Contains a copy of the original data and cleaned dataset.

---

## EDA
The subdirectory contains analysis analyzing histograms of the inputs, pair plots, and a PCA decomposition. In addition data cleaning is performed.

---

## Clustering
As certain input variables showed clear distinct gaussian distributions for each category of "Air Quality", these features were considered as input into a gaussian mixture model (GMM) to cluster the different categories of air quality. Results show high accuracy and that the cluster means are accurately at the means of the different gaussian distributions for the different variables.

---

## Classification
This subdirectory contains analysis of how different ML models are able to classify the air quality data. Models used include logistic regression, linear SVM (high performance, so other kernels not needed), random forest, XGBoost, and a neural network.

---

## XAI
This subdirectory looks into explaining the machine learning techniques and how the inputs to the models affect them. Feature importance from random forest and XGBoost are considered as well as shap values and permutation importance.
