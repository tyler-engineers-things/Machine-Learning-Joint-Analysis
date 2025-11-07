import sklearn
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

CSV_Path = "Hexapod_One_Joint.csv"
TARGET_COL = "Label"
FEARURE_COLS = ["X","Y","Z","slop(x)","slop(x/z)","slop(1/z)","slop(1/x)","slop(z/x)","slop(z)"]

df = pd.read_csv(CSV_Path)

# Separate features and target
feature_cols = df.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number]).columns.tolist()
X = df[feature_cols].copy()
y = df[TARGET_COL].astype(int)

#1 Missing Valyes
# Replace NaN entries by values estimated from nearby samples
imputer = KNNImputer(n_neighbors=3, weights="distance")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

#2 Removing Outliers
# Compute z-scores for each feature
z_scores = np.abs((X_imputed - X_imputed.mean()) / X_imputed.std())
# Keep rows where all features are within 3 std deviations
X_cleaned = X_imputed[(z_scores < 3).all(axis=1)]
y_cleaned = y.loc[X_cleaned.index].reset_index(drop=True)
X_cleaned = X_cleaned.reset_index(drop=True)

#3 Scale Features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_cleaned), columns=feature_cols)

# ---- RESULT ----
print("Original shape:", X.shape)
print("After cleaning:", X_scaled.shape)
print("Missing values remaining:", X_scaled.isna().sum().sum())
print("Preview of cleaned, scaled data:")
print(X_scaled.head())

cleaned_df = pd.concat([X_scaled, y_cleaned], axis=1)
cleaned_df.to_csv("hexapod_data_cleaned.csv", index=False)