# =========================
# Imports
# =========================
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =========================
# Utilidades
# =========================
def evaluate_model(y_true, y_pred, model_name, split_name=""):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    tag  = f" [{split_name}]" if split_name else ""
    print(f"\n{model_name}{tag} -> RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}

def plot_actual_vs_pred(y_true, y_pred, title="Actual vs Predicted"):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    lo, hi = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    plt.plot([lo, hi], [lo, hi], 'r--', lw=2)
    plt.xlabel('Actual Grades'); plt.ylabel('Predicted Grades')
    plt.title(title)
    plt.tight_layout(); plt.show(); plt.close()

# =========================
# Cargar dataset
# =========================
df = pd.read_csv('data.csv')

# =========================
# EDA breve (opcional)
# =========================
plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix'); plt.tight_layout(); plt.show(); plt.close()

# =========================
# Features / Target
# =========================
# Usamos 'Grades' como target y todo lo demás como features.
# Si existe 'student_id', se eliminará automáticamente por no ser útil.
X = df.drop(columns=['Grades', 'student_id'], errors='ignore')
y = df['Grades']

# =========================
# Train / Validation / Test
# =========================
# 1) Separamos TEST (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
# 2) De lo que queda, separamos VALIDATION (20% de 80% = 16% del total)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.20, random_state=42
)

print(f"Split sizes -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# =========================
# Escalado (ajustar SOLO con train)
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# =========================
# Modelos (solo 2)
# =========================
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(
        n_estimators=200,       # puedes usar 100 si prefieres
        max_depth=None,         # sin límite; ajusta si hay sobreajuste
        random_state=42,
        n_jobs=-1               # usa todos los cores disponibles
    )
}

# =========================
# Entrenamiento y evaluación (Val y Test)
# =========================
results = {}  # {model_name: {"val": {...}, "test": {...}}}
cv_summary = {}

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.fit(X_train_scaled, y_train)

    # Validación
    y_val_pred  = model.predict(X_val_scaled)
    res_val = evaluate_model(y_val, y_val_pred, name, split_name="Validation")

    # Test
    y_test_pred = model.predict(X_test_scaled)
    res_test = evaluate_model(y_test, y_test_pred, name, split_name="Test")

    # CV en train para estabilidad (opcional)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    cv_summary[name] = (cv_scores.mean(), cv_scores.std())
    print(f"CV R2 (5-fold) -> mean: {cv_scores.mean():.4f} | +/- {cv_scores.std()*2:.4f}")

    results[name] = {"val": res_val, "test": res_test}

# =========================
# Comparación R² (Validación)
# =========================
model_names = list(models.keys())
r2_val_scores  = [results[n]["val"]["r2"]  for n in model_names]
r2_test_scores = [results[n]["test"]["r2"] for n in model_names]

plt.figure(figsize=(10, 6))
plt.bar(model_names, r2_val_scores)
plt.title('Model Comparison - R2 (Validation)')
plt.ylabel('R2 Score'); plt.ylim(0, 1.05)
plt.xticks(rotation=15); plt.tight_layout(); plt.show(); plt.close()

# =========================
# Elegir mejor por Validación y graficar scatters
# =========================
best_idx = int(np.argmax(r2_val_scores))
best_model_name = model_names[best_idx]
best_model = models[best_model_name]

# Scatters Val y Test para el mejor
y_val_pred_best  = best_model.predict(X_val_scaled)
y_test_pred_best = best_model.predict(X_test_scaled)

plot_actual_vs_pred(y_val,  y_val_pred_best,
                    title=f'Actual vs Predicted (Validation) - {best_model_name}')
plot_actual_vs_pred(y_test, y_test_pred_best,
                    title=f'Actual vs Predicted (Test) - {best_model_name}')

# =========================
# Feature importance (solo RF)
# =========================
if 'Random Forest' in models:
    rf = models['Random Forest']
    # Asegúrate de que el RF esté entrenado (lo está)
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importances)
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout(); plt.show(); plt.close()

# =========================
# Resumen final y guardado
# =========================
print("\n" + "="*60)
print("RESUMEN FINAL (R² / RMSE / MAE)")
print("="*60)
for name in model_names:
    v = results[name]["val"]
    t = results[name]["test"]
    print(f"{name:>18} | VAL  R²={v['r2']:.4f} RMSE={v['rmse']:.2f} MAE={v['mae']:.2f} "
          f"| TEST R²={t['r2']:.4f} RMSE={t['rmse']:.2f} MAE={t['mae']:.2f}")
print("="*60)
print(f"🏆 Mejor por Validación: {best_model_name} | R² Val = {r2_val_scores[best_idx]:.4f}")

# Guardar mejor modelo + scaler
try:
    import joblib
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("✅ Models saved: best_model.pkl, scaler.pkl")
except Exception as e:
    print(f"⚠️ Could not save models: {e}")
