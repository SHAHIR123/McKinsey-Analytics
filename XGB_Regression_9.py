import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PolynomialFeatures

# Load data set
train = pd.read_csv("/home/shahir/myfolder/AV/Mckinsey_healthcare/train_ajEneEa.csv")
test = pd.read_csv("/home/shahir/myfolder/AV/Mckinsey_healthcare/test_v2akXPA.csv")
sample_submission = pd.read_csv("/home/shahir/myfolder/AV/Mckinsey_healthcare/sample_submission_1.csv")

# FIll missing values in the data set

df_train = train.fillna(method = 'bfill', axis=0).fillna("0")
df_test = test.fillna(method = 'bfill', axis=0).fillna("0")

#Encoding categorical data
le = LabelEncoder()

le.fit(np.hstack([df_train.smoking_status, df_test.smoking_status]))
df_train.smoking_status = le.transform(df_train.smoking_status)
df_test.smoking_status = le.transform(df_test.smoking_status)
del le

# Set target variable and predictors
y = df_train["stroke"]
predictors = df_train[[ "age", "hypertension", "heart_disease","avg_glucose_level","smoking_status"]]
test_predictors = df_test[[ "age", "hypertension", "heart_disease","avg_glucose_level","smoking_status"]]

poly = PolynomialFeatures(2)
poly.fit_transform(predictors)
poly.transform(test_predictors)

# Trains split for model evaluation
X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(predictors), y, test_size=0.33, random_state=42)

# Make and evaluate model
model = XGBRegressor()
model.fit(X_train,y_train)
prediction = model.predict(X_test)
score = roc_auc_score(y_test, prediction, average='macro', sample_weight=None)
print(score)

# Fit model
model.fit(predictors, y)

# Predict on test data
prediction = model.predict(test_predictors)

# Create submission file
my_submission = pd.DataFrame({'id': test.id, 'stroke': prediction})
my_submission.to_csv('XGB_Regression_submission_9.csv', index=False)