import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, log_loss, roc_auc_score)

df = pd.read_csv("hexapod_data_cleaned.csv")
X = df.drop(columns=["Label"])
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True))
])

svm_pipeline.fit(X_train, y_train)

y_pred = svm_pipeline.predict(X_test)
y_proba = svm_pipeline.predict_proba(X_test)

print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nLog Loss: {log_loss(y_test, y_proba):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba, multi_class='ovr'):.4f}")

clf = svm_pipeline.named_steps["clf"]
feature_names = X.columns

if hasattr(clf, "support_"):
    support_vectors = clf.support_vectors_
    support_indices = clf.support_
    print(f"\nNumber of support vectors: {len(support_indices)}")
    print("Support vector indices:", support_indices)
else:
    print("The SVC model does not provide support vector information.")