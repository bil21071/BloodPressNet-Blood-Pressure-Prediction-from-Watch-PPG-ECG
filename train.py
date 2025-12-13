import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from xgboost import plot_importance
import numpy as np

df = pd.read_csv("features_vm_3.csv")
df = df.replace([np.inf, -np.inf], np.nan)
# cols = ['notch_amp', 'reflective_idx', 'crest_time', 'width_25', 'width_50']
# df = df[~(df[cols] == 0).any(axis=1)] # improves accuarracy
df = df.dropna()

# Features = all except SBP, DBP
X = df.drop(columns=["SBP", "DBP"])
y_sbp = df["SBP"]
y_dbp = df["DBP"]

# TRIMMING (quanitles)
# lower_q = 0.05
# upper_q = 0.95

# quantiles = X.quantile([lower_q, upper_q])

# mask = ((X >= quantiles.loc[lower_q]) & (X <= quantiles.loc[upper_q])).all(axis=1)

# X = X[mask]
# y_sbp = y_sbp[mask]
# y_dbp = y_dbp[mask]


# split Train/Test
X_train, X_test, y_sbp_train, y_sbp_test = train_test_split(X, y_sbp, test_size=0.2, random_state=42)
_, _, y_dbp_train, y_dbp_test = train_test_split(X, y_dbp, test_size=0.2, random_state=42)

# SBP model
model_sbp = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model_sbp.fit(X_train, y_sbp_train)

# DBP model
model_dbp = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model_dbp.fit(X_train, y_dbp_train)

# Predictions
y_sbp_pred = model_sbp.predict(X_test)
y_dbp_pred = model_dbp.predict(X_test)

y_true = y_sbp_test
y_pred = y_sbp_pred

corr = df.corr()
corr[["SBP", "DBP"]].sort_values(by="SBP", ascending=False)
# print(corr)



plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.6, label="Samples")

# regression line (fit between y_true and y_pred)
m, b = np.polyfit(y_true, y_pred, 1)
plt.plot(y_true, m*y_true + b, color="red", label=f"Fit: y={m:.2f}x+{b:.2f}")

# perfect prediction line
plt.plot([y_true.min(), y_true.max()],
         [y_true.min(), y_true.max()],
         color="black", linestyle="--", label="Ideal")

plt.xlabel("True SBP")
plt.ylabel("Predicted SBP")
plt.title("SBP Prediction: True vs Predicted")
plt.legend()
plt.grid(True)
plt.savefig("sbp_bilal.png")
plt.close()

# Metrics
print("SBP MAE:", mean_absolute_error(y_sbp_test, y_sbp_pred))
print("SBP R²:", r2_score(y_sbp_test, y_sbp_pred))
print("DBP MAE:", mean_absolute_error(y_dbp_test, y_dbp_pred))
print("DBP R²:", r2_score(y_dbp_test, y_dbp_pred))

# feature importance
plt.figure(figsize=(10,6))
plot_importance(model_sbp, max_num_features=10, importance_type="gain")
plt.title("Top 10 Features for SBP Prediction")
plt.savefig("top_features_bilal.png")
plt.close()



# Hyperparameter
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [200, 500, 1000],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

xgb = XGBRegressor(random_state=42, n_jobs=-1)
grid = GridSearchCV(xgb, param_grid, cv=3, scoring="r2", verbose=1, n_jobs=-1)
grid.fit(X, y_sbp)

print("Best params:", grid.best_params_)
print("Best CV R²:", grid.best_score_)


from sklearn.dummy import DummyRegressor

# --- Baseline model for SBP ---
dummy_sbp = DummyRegressor(strategy="mean")
dummy_sbp.fit(X_train, y_sbp_train)
y_sbp_dummy_pred = dummy_sbp.predict(X_test)

print("=== Baseline (Dummy) SBP ===")
print("MAE:", mean_absolute_error(y_sbp_test, y_sbp_dummy_pred))
print("R² :", r2_score(y_sbp_test, y_sbp_dummy_pred))

# --- Baseline model for DBP ---
dummy_dbp = DummyRegressor(strategy="mean")
dummy_dbp.fit(X_train, y_dbp_train)
y_dbp_dummy_pred = dummy_dbp.predict(X_test)

print("\n=== Baseline (Dummy) DBP ===")
print("MAE:", mean_absolute_error(y_dbp_test, y_dbp_dummy_pred))
print("R² :", r2_score(y_dbp_test, y_dbp_dummy_pred))
