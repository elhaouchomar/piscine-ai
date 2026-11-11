# Forest Cover Type Prediction

## Project Description

This project uses cartographic variables (elevation, slope, distances to hydrology, roads, fire points, hillshade readings, soil types and wilderness areas) to classify forest cover into 7 categories. It was developed for an environmental conservation agency to support forest management decision-making.

The dataset is based on the UCI Covertype dataset. A machine learning pipeline was built covering EDA, feature engineering, model selection via cross-validated grid search, and final prediction on a held-out test set.

---

## Results

| Metric | Value |
|---|---|
| Best model | GradientBoostingClassifier |
| Train accuracy (Train set 0) | **0.9570** ✓ (< 0.98) |
| CV accuracy (5-fold, Train set 1) | **0.9236** |
| Test accuracy (Test set 1) | **0.9298** |
| Test accuracy (Test set 0 – final) | **0.6779** ✓ (> 0.65) |


---

## How to Run from a Fresh Environment

### 1. Install dependencies

It is recommended to use a virtual environment:

```bash
# Using pip
pip install -r requirements.txt
```

### 2. Run feature engineering (standalone check)

```bash
python scripts/preprocessing_feature_engineering.py
```

### 3. Run model selection (trains all 5 models, saves best)

```bash
python scripts/model_selection.py
```

This produces:
- `results/best_model.pkl`
- `results/confusion_matrix_heatmap.png`
- `results/learning_curve_best_model.png`

### 4. Predict on test set

```bash
python scripts/predict.py
```

This produces:
- `results/test_predictions.csv`

---

## Data Splitting Strategy

```
TRAIN FILE (0)  [train.csv]
├── Train (1)  →  80% of train.csv
│       Fold 0: Train / Validation
│       Fold 1: Train / Validation
│       Fold 2: Train / Validation
│       Fold 3: Train / Validation
│       Fold 4: Train / Validation
└── Test (1)   →  20% of train.csv  (held-out during CV)

TEST FILE (0)   [test.csv]  ← used only in predict.py
```

---

## Python Scripts Summary

### `scripts/preprocessing_feature_engineering.py`

Loads raw CSV data and engineers 5 new features:

| Feature | Formula / Logic | Rationale |
|---|---|---|
| `Distance_To_Hydrology` | `sqrt(H_dist² + V_dist²)` | True Euclidean proximity to water |
| `Fire_Road_Difference` | `H_fire – H_road` | Relative remoteness from roads vs fire points |
| `Mean_Hillshade` | `(shade_9am + shade_noon + shade_3pm) / 3` | Average daily sun exposure |
| `Elev_Hydro_Interaction` | `Elevation × Distance_To_Hydrology` | High-elevation water proximity matters differently per cover type |
| `High_Elevation_Flag` | `1 if Elevation > 3000 else 0` | Binary separator for Krummholz (type 7) vs lower types |

Public API: `preprocess(path)` → returns `(X, y)` ready for modelling.

---

### `scripts/model_selection.py`

Implements the full model selection pipeline:

1. Loads and feature-engineers training data via `preprocess()`
2. Splits into **Train (1) / Test (1)** at 80/20 (ratio < 33%)
3. Defines **5 model pipelines** with preprocessing where needed:
   - **GradientBoosting** – no scaling needed
   - **RandomForest** – no scaling needed
   - **KNN** – `StandardScaler` + KNeighborsClassifier
   - **SVM** – `StandardScaler` + LinearSVC
   - **LogisticRegression** – `StandardScaler` + LogisticRegression
4. Runs **GridSearchCV with 5-fold StratifiedKFold** on each model
5. Selects best model by CV accuracy (GradientBoosting: CV=0.9482)
6. Validates: train accuracy < 0.98, prints confusion matrix as DataFrame
7. Saves confusion matrix heatmap, learning curve plot, and `best_model.pkl`

---

### `scripts/predict.py`

1. Loads `results/best_model.pkl`
2. Loads and feature-engineers `data/test.csv` via `preprocess()`
3. Predicts and prints test set (0) accuracy
4. Saves predictions to `results/test_predictions.csv`

---

## Project Structure

```
project/
│   README.md
│   requirements.txt
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── covtype.info
│
├── notebook/
│   └── EDA.ipynb
│
├── scripts/
│   ├── preprocessing_feature_engineering.py
│   ├── model_selection.py
│   └── predict.py
│
└── results/
    ├── confusion_matrix_heatmap.png
    ├── learning_curve_best_model.png
    ├── test_predictions.csv
    └── best_model.pkl
```
