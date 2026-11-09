"""
Personalized Learning System - Student Performance Classification
Technique : Classification (single data-mining technique)
Models    : Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
Dataset   : Kaggle - lainguyn123/student-performance-factors (StudentPerformanceFactors.csv)
"""
import json, sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             classification_report, ConfusionMatrixDisplay)

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/StudentPerformanceFactors.csv"
RANDOM_STATE = 42
LABELS = ["Needs Support", "On Track", "Advanced"]

# ---------------------------------------------------------------- 1. Load
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0], "\n")

# ---------------------------------------------------------------- 2. Target
# Exam_Score -> 3 learner levels using quantile cut-offs (balanced classes)
q1, q2 = df["Exam_Score"].quantile([1/3, 2/3])
df["Level"] = pd.cut(df["Exam_Score"], bins=[-np.inf, q1, q2, np.inf], labels=LABELS)
print("Level distribution:\n", df["Level"].value_counts(), "\n")

X = df.drop(columns=["Exam_Score", "Level"])
y = df["Level"].astype(str)
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

# ---------------------------------------------------------------- 3. Preprocessing
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

# ---------------------------------------------------------------- 4. Three+ models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

rows, fitted = [], {}
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    cv_acc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy").mean()
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    rows.append({
        "Model": name,
        "CV Accuracy": cv_acc,
        "Test Accuracy": accuracy_score(y_te, pred),
        "Precision (macro)": precision_score(y_te, pred, average="macro"),
        "Recall (macro)": recall_score(y_te, pred, average="macro"),
        "F1 (macro)": f1_score(y_te, pred, average="macro"),
    })
    fitted[name] = (pipe, pred)
    print(f"=== {name} ===")
    print(classification_report(y_te, pred, labels=LABELS), "\n")

results = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False).round(4)
results.to_csv("model_comparison.csv", index=False)
print(results.to_string(index=False))

# ---------------------------------------------------------------- 5. Charts
ax = results.set_index("Model")[["Test Accuracy", "F1 (macro)"]].plot(
    kind="bar", figsize=(8, 5), rot=15)
ax.set_ylim(0, 1); ax.set_title("Model comparison"); plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150); plt.close()

best_name = results.iloc[0]["Model"]
best_pipe, best_pred = fitted[best_name]
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_te, best_pred, labels=LABELS, ax=ax, colorbar=False)
ax.set_title(f"Confusion matrix - {best_name}"); plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150); plt.close()

# Feature importance (best tree model if available, else RF)
rf_pipe = fitted["Random Forest"][0]
names = rf_pipe.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf_pipe.named_steps["clf"].feature_importances_, index=names).sort_values()[-12:]
imp.plot(kind="barh", figsize=(7, 5), title="Top features (Random Forest)")
plt.tight_layout(); plt.savefig("feature_importance.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 6. Save best model + UI metadata
joblib.dump(best_pipe, "best_model.joblib")
meta = {
    "best_model": best_name,
    "labels": LABELS,
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "num_ranges": {c: [float(X[c].min()), float(X[c].max()), float(X[c].median())] for c in num_cols},
    "cat_options": {c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols},
    "cat_default": {c: X[c].mode()[0] for c in cat_cols},
}
json.dump(meta, open("model_meta.json", "w"), indent=2)
print(f"\nSaved best model: {best_name}")
