# =====================
# Import necessary libraries
# =====================
import pandas as pd
import numpy as np
import pickle
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
 
# =====================
# Load dataset
# =====================
df = pd.read_csv("HR-Employee-Attrition.csv")

print(df)

# =====================
# Drop useless columns
# =====================
df.drop([
    'EmployeeCount',
    'EmployeeNumber',
    'Over18',
    'StandardHours'
], axis=1, inplace=True)

# =====================
# Target and features
# =====================
X = df.drop('Attrition', axis=1)
y = df['Attrition'].map({'Yes': 1, 'No': 0})

# =====================
# Column split
# =====================
num_cols = X.select_dtypes(exclude='object').columns
cat_cols = X.select_dtypes(include='object').columns

# =====================
# Preprocessing
# =====================
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# =====================
# Train-test split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scale_pos_weight = (y_train.value_counts()[0] / y_train.value_counts()[1])

# =====================
# XGBoost Model
# =====================
xgb_model = XGBClassifier(
        n_estimators=189,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42
    )

# =====================
# Full Pipeline
# =====================
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])

xgb_pipeline.fit(X_train, y_train)

# =====================
# Evaluation
# =====================
y_pred = xgb_pipeline.predict(X_test)
y_proba = xgb_pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

# =====================
# Save model
# =====================
with open('xgb_model.pkl', 'wb') as file:
    pickle.dump(xgb_pipeline, file)
print("Model saved as xgb_model.pkl")