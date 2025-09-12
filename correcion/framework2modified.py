# =========================
# Imports
# =========================
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =========================
# Cargar dataset
# =========================
df = pd.read_csv('data.csv')


# =========================
# Features / Target
# =========================
X = df.drop('Grades', axis=1)
y = df['Grades']

# =========================
# Split y escalado
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# =========================
# Evaluación de modelos
# =========================
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} Performance Metrics:")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f}")
    return rmse, mae, r2

# =========================
# SOLO 2 MODELOS
# =========================
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = evaluate_model(y_test, y_pred, name)

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"Cross-validation R2 scores: {cv_scores}")
    print(f"Mean CV R2 score: {cv_scores.mean():.2f} (+/- {cv_scores.std()*2:.2f})")

# =========================
# Feature importance (RF)
# =========================
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance (Random Forest)')
plt.show()

# =========================
# Comparación R²
# =========================
model_names = list(models.keys())
r2_scores = [results[name][2] for name in model_names]

plt.figure(figsize=(10, 6))
plt.bar(model_names, r2_scores)
plt.title('Model Comparison - R2 Scores')
plt.xticks(rotation=15)
plt.ylabel('R2 Score')
plt.tight_layout()
plt.show()

# =========================
# Scatter mejor modelo
# =========================
best_model_name = model_names[np.argmax(r2_scores)]
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Grades')
plt.ylabel('Predicted Grades')
plt.title(f'Actual vs Predicted Grades ({best_model_name})')
plt.tight_layout()
plt.show()

# =========================
# Guardado
# =========================
import joblib
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f"\nBest performing model: {best_model_name}")
