import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    learning_curve,
)
from sklearn.metrics import accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from preprocessing_feature_engineering import preprocess

TRAIN_PATH  = os.path.join(DATA_DIR, "train.csv")
MODEL_PATH  = os.path.join(RESULTS_DIR, "best_model.pkl")
CM_PLOT     = os.path.join(RESULTS_DIR, "confusion_matrix_heatmap.png")
LC_PLOT     = os.path.join(RESULTS_DIR, "learning_curve_best_model.png")

print("=" * 60)
print("Loading and engineering features …")
X_all, y_all = preprocess(TRAIN_PATH)
print(f"  Full train file shape : {X_all.shape}")

# Train (1) / Test (1) split  (80 / 20, ratio < 33 %)
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
)
print(f"  Train (1) size : {X_train1.shape[0]}")
print(f"  Test  (1) size : {X_test1.shape[0]}")

CV        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
N_JOBS    = -1
VERBOSE   = 2

pipelines = {
    "GradientBoosting": (
        Pipeline([("clf", GradientBoostingClassifier(random_state=42))]),
        {
            "clf__n_estimators":  [100],
            "clf__max_depth":     [5],
        },
    ),
    "RandomForest": (
        Pipeline([("clf", RandomForestClassifier(random_state=42, n_jobs=N_JOBS))]),
        {
            "clf__n_estimators": [100],
            "clf__max_depth":    [12],
        },
    ),
    "KNN": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(n_jobs=N_JOBS)),
        ]),
        {
            "clf__n_neighbors": [5],
        },
    ),
    "SVM": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(random_state=42, cache_size=500)),
        ]),
        {
            "clf__C":      [1],
        },
    ),
    "LogisticRegression": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, random_state=42, n_jobs=N_JOBS)),
        ]),
        {
            "clf__C":      [1],
        },
    ),
}

print("\n" + "=" * 60)
print("Running GridSearchCV for each model …\n")

results = {}
for name, (pipe, param_grid) in pipelines.items():
    print(f" {name}")
    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=CV,
        scoring="accuracy",
        n_jobs=N_JOBS,
        verbose=VERBOSE,
        refit=True,
    )
    gs.fit(X_train1, y_train1)
    cv_score  = gs.best_score_
    train_acc = accuracy_score(y_train1, gs.best_estimator_.predict(X_train1))
    test_acc  = accuracy_score(y_test1,  gs.best_estimator_.predict(X_test1))
    results[name] = {
        "gs":        gs,
        "cv_acc":    cv_score,
        "train_acc": train_acc,
        "test_acc":  test_acc,
        "params":    gs.best_params_,
    }
    print(f"     Best CV acc    : {cv_score:.4f}")
    print(f"     Train (1) acc  : {train_acc:.4f}")
    print(f"     Test  (1) acc  : {test_acc:.4f}")
    print(f"     Best params    : {gs.best_params_}\n")

# Select best model (highest CV accuracy)
best_name = max(results, key=lambda n: results[n]["cv_acc"])
best_info = results[best_name]
best_model = best_info["gs"].best_estimator_

# Final evaluation of the best model
best_train0_acc = accuracy_score(y_all, best_model.predict(X_all))
print("=" * 60)
print(f"BEST MODEL : {best_name}")
print(f"  CV accuracy (Train 1)     : {best_info['cv_acc']:.4f}")
print(f"  Test accuracy (Test 1)     : {best_info['test_acc']:.4f}")
print(f"  Full Train accuracy (Train 0) : {best_train0_acc:.4f}  (must be < 0.98)")

assert best_train0_acc < 0.98, (
    f"Train set (0) accuracy {best_train0_acc:.4f} >= 0.98 — reduce model complexity!"
)

# Confusion matrix as DataFrame
y_pred = best_model.predict(X_test1)
classes = sorted(y_all.unique())
cm = confusion_matrix(y_test1, y_pred, labels=classes)
cm_df = pd.DataFrame(
    cm,
    index=pd.Index(classes, name="True label"),
    columns=pd.Index(classes, name="Predicted label"),
)
print("\nConfusion matrix (Test (1)):")
print(cm_df)

# Save heatmap
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
fig.colorbar(im, ax=ax)
ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
ax.set_title(f"Confusion Matrix – {best_name}")
thresh = cm.max() / 2.0
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.savefig(CM_PLOT, dpi=150)
plt.close()
print(f"\nConfusion matrix heatmap saved {CM_PLOT}")

# Learning curve
print("\nGenerating learning curve …")
train_sizes, train_scores, val_scores = learning_curve(
    best_model,
    X_train1, y_train1,
    cv=CV,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 8),
    n_jobs=N_JOBS,
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(train_sizes, train_mean, "o-", color="royalblue",  label="Training accuracy")
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="royalblue")
ax.plot(train_sizes, val_mean,   "o-", color="green",      label="Validation accuracy (CV)")
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="green")
ax.set_xlabel("Training set size"); ax.set_ylabel("Accuracy")
ax.set_title(f"Learning Curve – {best_name}")
ax.legend(loc="lower right"); ax.grid(True)
plt.tight_layout()
plt.savefig(LC_PLOT, dpi=150)
plt.close()
print(f"Learning curve saved {LC_PLOT}")

# Save model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)
print(f"\nBest model saved {MODEL_PATH}")
print("=" * 60)
print("model_selection.py complete.")
