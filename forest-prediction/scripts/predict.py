import os
import sys
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")

sys.path.insert(0, SCRIPT_DIR)
from preprocessing_feature_engineering import preprocess

TEST_PATH    = os.path.join(DATA_DIR,    "test.csv")
MODEL_PATH   = os.path.join(RESULTS_DIR, "best_model.pkl")
PRED_PATH    = os.path.join(RESULTS_DIR, "test_predictions.csv")

print("Loading model from", MODEL_PATH)
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print("Model loaded:", type(model).__name__)

print("Loading test data from", TEST_PATH)
X_test, y_test = preprocess(TEST_PATH)
print(f"Test set shape : {X_test.shape}")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest set (0) accuracy : {acc:.4f}")
assert acc > 0.65, f"Test accuracy {acc:.4f} is below the required 0.65 threshold!"
print("Accuracy > 0.65 requirement satisfied")

os.makedirs(RESULTS_DIR, exist_ok=True)
pred_df = pd.DataFrame({"True_Label": y_test.values, "Predicted_Label": y_pred})
pred_df.to_csv(PRED_PATH, index=False)
print(f"\nPredictions saved: {PRED_PATH}")
