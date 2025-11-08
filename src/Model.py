import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, log_loss, roc_auc_score)


df = pd.read_csv("hexapod_data_cleaned.csv")

 
X = df.drop(columns=["Label"])
y = df["Label"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

log_reg_pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500, C=1.0, n_jobs=1))])

log_reg_pipeline.fit(X_train, y_train)

y_pred = log_reg_pipeline.predict(X_test)
y_proba = log_reg_pipeline.predict_proba(X_test)

print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nLog Loss:")
print(log_loss(y_test, y_proba))


try:
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
    print("\nMacro ROC-AUC (OvR):")
    print(auc)
except ValueError as e:
    print("\nROC-AUC could not be computed:", e)


clf = log_reg_pipeline.named_steps["clf"]
feature_names = X.columns

for class_idx, class_label in enumerate(clf.classes_):
    coefs = clf.coef_[class_idx]
    top_idx = coefs.argsort()[::-1][:5]
    print(f"\nTop features for class {class_label}:")
    for i in top_idx:
        print(f"  {feature_names[i]}: {coefs[i]:.4f}")

