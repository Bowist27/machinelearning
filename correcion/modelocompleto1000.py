import csv
import random
import math
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# =========================
# TUS FUNCIONES (idénticas)
# =========================
def hyp(x, theta, b):
    # Si x es una lista de listas (matriz), procesar cada fila
    if isinstance(x[0], list):
        results = []
        for sample in x:
            y_hat = 0
            for i in range(len(sample)):
                y_hat += sample[i] * theta[i]
            y_hat += b
            results.append(y_hat)
        return results
    else:   # Si x es una sola muestra, procesarla normalmente
        y_hat = 0
        for i in range(len(x)):
            y_hat += x[i] * theta[i]
        y_hat += b
        return y_hat

def mse_cost_from_preds(preds, Y):
    m = len(Y)
    errors = [preds[i] - Y[i] for i in range(m)]
    sq_errors = [e**2 for e in errors]
    total = sum(sq_errors)
    cost = total / (2*m)  # J = (1/(2m)) Σ (h - y)^2
    return cost, errors

def grad_Descent(X, Y, theta, b, alpha):
    preds = hyp(X, theta, b)
    cost, errors = mse_cost_from_preds(preds, Y)

    m, n = len(Y), len(theta)
    db = sum(errors) / m

    dtheta = []
    for j in range(n):
        s = 0.0
        for i in range(m):
            s += errors[i] * X[i][j]
        s /= m
        dtheta.append(s)

    theta_new = [theta[j] - alpha * dtheta[j] for j in range(n)]
    b_new = b - alpha * db

    preds_new = hyp(X, theta_new, b_new)
    cost_new, _ = mse_cost_from_preds(preds_new, Y)
    return theta_new, b_new, cost_new

# =========================
# UTILIDADES
# =========================

# Encabezados esperados para data.csv (según tu captura)
FEATURES = ["Socioeconomic Score", "Study Hours", "Sleep Hours", "Attendance (%)"]
TARGET   = "Grades"

# Aliases por si el CSV trae mayúsculas/espacios diferentes
HEADER_ALIASES = {
    "socioeconomic score": "Socioeconomic Score",
    "study hours": "Study Hours",
    "sleep hours": "Sleep Hours",
    "attendance (%)": "Attendance (%)",
    "attendance %": "Attendance (%)",
    "grades": "Grades"
}

def _normalize_header(h: str) -> str:
    return h.strip().lower().replace("\ufeff", "")

def _remap_headers(fieldnames: List[str]) -> Dict[str, str]:
    mapping = {}
    for raw in fieldnames:
        key = _normalize_header(raw)
        std = HEADER_ALIASES.get(key, raw)
        mapping[raw] = std
    return mapping

def zscore_fit(X: List[List[float]]) -> Tuple[List[float], List[float]]:
    """Calcula media y std (por columna) para estandarizar features."""
    n_feat = len(X[0])
    means, stds = [], []
    for j in range(n_feat):
        col = [row[j] for row in X]
        mu = sum(col) / len(col)
        var = sum((v - mu)**2 for v in col) / max(1, (len(col)-1))
        sd = math.sqrt(var) if var > 0 else 1.0
        means.append(mu)
        stds.append(sd)
    return means, stds

def zscore_transform(X, means, stds):
    Xs = []
    for row in X:
        Xs.append([(row[j] - means[j]) / stds[j] for j in range(len(row))])
    return Xs

def train(X, Y, theta=None, b=None, alpha=1e-2, epochs=2000, tol=None, verbose=False):
    """
    BGD puro: cada epoch hace EXACTAMENTE un grad_step sobre todo el conjunto.
    Devuelve también el historial de coste para graficar.
    """
    n = len(X[0])
    if theta is None:
        theta = [0.0] * n
    if b is None:
        b = 0.0

    prev_cost = None
    history = []
    for ep in range(1, epochs + 1):
        theta, b, cost = grad_Descent(X, Y, theta, b, alpha)
        history.append(cost)
        if verbose and (ep == 1 or ep % max(1, epochs // 10) == 0 or ep == epochs):
            print(f"epoch {ep:4d} | cost={cost:.6f} | theta={theta} | b={b:.6f}")
        if tol is not None and prev_cost is not None and abs(prev_cost - cost) < tol:
            if verbose:
                print(f"Paro temprano por tol={tol} en epoch {ep}.")
            break
        prev_cost = cost
    return theta, b, history

def read_csv_numeric(path: str) -> Tuple[List[str], List[Dict[str, float]]]:
    """
    Lee CSV y devuelve headers y filas numéricas.
    Convierte a float cuando puede y remapea encabezados a los esperados.
    """
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        original_headers = reader.fieldnames
        header_map = _remap_headers(original_headers)

        for r in reader:
            row_dict = {}
            for raw_k, v in r.items():
                k = header_map[raw_k]
                if k == 'student_id':
                    continue
                try:
                    if v not in (None, "", "NA", "NaN"):
                        row_dict[k] = float(v)
                    else:
                        row_dict[k] = float("nan")
                except ValueError:
                    # ignora columnas no numéricas
                    continue
            rows.append(row_dict)

    numeric_headers = [HEADER_ALIASES.get(_normalize_header(h), h) for h in original_headers if HEADER_ALIASES.get(_normalize_header(h), h) != 'student_id']
    return numeric_headers, rows

def select_xy(rows, feature_names: List[str], target_name: str):
    X, Y = [], []
    for r in rows:
        try:
            feats = [r[name] for name in feature_names]
            if any(math.isnan(v) for v in feats) or math.isnan(r[target_name]):
                continue
            X.append(feats)
            Y.append(r[target_name])
        except KeyError:
            continue
    return X, Y

def train_val_test_split(X, Y, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8, "Las proporciones deben sumar 1."
    n = len(Y)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)

    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val  # asegura que todo sume n

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    Xtr = [X[i] for i in train_idx]; Ytr = [Y[i] for i in train_idx]
    Xva = [X[i] for i in val_idx];   Yva = [Y[i] for i in val_idx]
    Xte = [X[i] for i in test_idx];  Yte = [Y[i] for i in test_idx]
    return Xtr, Xva, Xte, Ytr, Yva, Yte

def mse_from_model(X, Y, theta, b):
    preds = hyp(X, theta, b)
    m = len(Y)
    return sum((preds[i] - Y[i])**2 for i in range(m)) / m

def r2_score(y_true, y_pred):
    """Coeficiente de determinación R^2."""
    n = len(y_true)
    if n == 0:
        return float('nan')
    y_bar = sum(y_true) / n
    ss_res = sum((y_true[i] - y_pred[i])**2 for i in range(n))
    ss_tot = sum((y - y_bar)**2 for y in y_true)
    if ss_tot == 0:
        return float('nan')  # R^2 indefinido si no hay varianza en y
    return 1.0 - ss_res / ss_tot

def r2_from_model(X, Y, theta, b):
    preds = hyp(X, theta, b)
    return r2_score(Y, preds)

# =========================
# HISTOGRAMAS CONSOLIDADOS
# =========================
def plot_all_histograms(
    Xtr, Xva, Xte, Ytr, Yva, Yte,
    feature_names, target_name,
    bins=20, suptitle="Distribuciones",
    train_color="#1f77b4", val_color="#ff7f0e", test_color="#2ca02c"
):
    """
    Dibuja TODOS los histogramas en una sola figura grande.
    Filas = variables (features + target)
    Columnas = Train | Val | Test
    """
    items = feature_names + [target_name]
    n_items = len(items)

    fig, axes = plt.subplots(n_items, 3, figsize=(12, 3*n_items), sharey='row')
    if n_items == 1:
        axes = [axes]

    fig.suptitle(suptitle, fontsize=14, y=1.01)
    col_titles = ['Train', 'Validation', 'Test']

    for i, name in enumerate(items):
        if i < len(feature_names):
            col_tr = [row[i] for row in Xtr]
            col_va = [row[i] for row in Xva]
            col_te = [row[i] for row in Xte]
        else:
            col_tr, col_va, col_te = Ytr, Yva, Yte

        axes[i][0].hist(col_tr, bins=bins, alpha=0.85, color=train_color)
        if i == 0:
            axes[i][0].set_title(f"{col_titles[0]} (n={len(col_tr)})")
        axes[i][0].set_ylabel(name, fontweight='bold'); axes[i][0].grid(True, alpha=0.3)

        axes[i][1].hist(col_va, bins=bins, alpha=0.85, color=val_color)
        if i == 0:
            axes[i][1].set_title(f"{col_titles[1]} (n={len(col_va)})")
        axes[i][1].grid(True, alpha=0.3)

        axes[i][2].hist(col_te, bins=bins, alpha=0.85, color=test_color)
        if i == 0:
            axes[i][2].set_title(f"{col_titles[2]} (n={len(col_te)})")
        axes[i][2].grid(True, alpha=0.3)

    for j in range(3):
        axes[-1][j].set_xlabel('Valor')

    plt.tight_layout()
    plt.show()
    plt.close(fig)

# =========================
# SCATTER ACTUAL vs PREDICHO
# =========================
def plot_actual_vs_pred(y_true, y_pred, title="Actual vs Predicted"):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    y_min, y_max = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    plt.plot([y_min, y_max], [y_min, y_max], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.tight_layout()
    plt.show()


# =========================
# EXPERIMENTO (único para data.csv)
# =========================
def run_experiment(csv_path: str, feature_names: List[str], target_name: str,
                   alpha=5e-2, epochs=4000, tol=1e-8, verbose=False, plot=True, title="",
                   hist_mode="raw"):
    # 1) Cargar datos
    headers, rows = read_csv_numeric(csv_path)

    # 2) Seleccionar X, Y
    X, Y = select_xy(rows, feature_names, target_name)
    if len(X) == 0:
        raise ValueError(f"No se pudieron construir datos con columnas {feature_names} -> {target_name}. Revisa los nombres.")

    # 3) Split 50/25/25
    Xtr, Xva, Xte, Ytr, Yva, Yte = train_val_test_split(
        X, Y, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=123
    )

    print(f"→ División: Train={len(Xtr)}, Val={len(Xva)}, Test={len(Xte)}")

    # 4) Estandarizar SOLO features usando estadísticas de TRAIN
    mu, sd = zscore_fit(Xtr)
    Xtr_s = zscore_transform(Xtr, mu, sd)
    Xva_s = zscore_transform(Xva, mu, sd)
    Xte_s = zscore_transform(Xte, mu, sd)

    # Histograma único (elige "raw" o "std")
    if plot and hist_mode == "raw":
        plot_all_histograms(
            Xtr, Xva, Xte, Ytr, Yva, Yte,
            feature_names, target_name,
            bins=20, suptitle=f"Datos RAW - {title or 'Experimento'}"
        )
    elif plot and hist_mode == "std":
        plot_all_histograms(
            Xtr_s, Xva_s, Xte_s, Ytr, Yva, Yte,
            feature_names, target_name,
            bins=20, suptitle=f"Datos ESTANDARIZADOS - {title or 'Experimento'}"
        )

    # 5) Entrenar con BGD SOLO con TRAIN
    theta0 = [0.0] * len(Xtr_s[0])
    b0 = 0.0
    theta, b, history = train(
        Xtr_s, Ytr, theta=theta0, b=b0, alpha=alpha, epochs=epochs, tol=tol, verbose=verbose
    )

    # 6) Evaluación en Train/Val/Test (MSE + R^2)
    mse_tr = mse_from_model(Xtr_s, Ytr, theta, b)
    mse_va = mse_from_model(Xva_s, Yva, theta, b)
    mse_te = mse_from_model(Xte_s, Yte, theta, b)

    r2_va = r2_from_model(Xva_s, Yva, theta, b)
    r2_te = r2_from_model(Xte_s, Yte, theta, b)

    # 7) Gráficas de dispersión Actual vs Predicho
    # Predicciones completas para Val y Test
    preds_va = hyp(Xva_s, theta, b)
    preds_te = hyp(Xte_s, theta, b)

    # Gráficas tipo "Actual vs Predicted"
    plot_actual_vs_pred(Yva, preds_va, title=f'Actual vs Predicted (Validación) - {title or "Experimento"}')
    plot_actual_vs_pred(Yte, preds_te, title=f'Actual vs Predicted (Test) - {title or "Experimento"}')


    print(f"\n== {title or 'Experimento'} ==")
    print(f"Features: {feature_names}")
    print(f"Theta (sobre features estandarizadas): {theta}")
    print(f"b: {b:.6f}")
    print(f"MSE Train: {mse_tr:.4f} | MSE Val: {mse_va:.4f} | MSE Test: {mse_te:.4f}")
    print(f"R^2  Val:  {r2_va:.4f} | R^2  Test: {r2_te:.4f}")

    # Predicciones de ejemplo (Test)
    sample = min(5, len(Xte_s))
    preds_sample = hyp(Xte_s[:sample], theta, b)
    print("\nPredicciones de ejemplo (Test):")
    for i in range(sample):
        print(f"  y_pred={preds_sample[i]:.2f} | y_real={Yte[i]:.2f} | x(estd)={Xte_s[i]}")

    # Predicciones de ejemplo (Validación)
    sample_val = min(5, len(Xva_s))
    preds_sample_val = hyp(Xva_s[:sample_val], theta, b)
    print("\nPredicciones de ejemplo (Validación):")
    for i in range(sample_val):
        print(f"  y_pred={preds_sample_val[i]:.2f} | y_real={Yva[i]:.2f} | x(estd)={Xva_s[i]}")

    # 7) Curva de aprendizaje (opcional)
    if plot:
        plt.figure(figsize=(7,5))
        plt.plot(range(1, len(history)+1), history)
        plt.xlabel("Epoch")
        plt.ylabel("Costo J")
        plt.title(f"Curva de coste - {title or 'Experimento'}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        plt.close()

    return {
        "theta": theta, "b": b,
        "mse_train": mse_tr, "mse_val": mse_va, "mse_test": mse_te,
        "r2_val": r2_va, "r2_test": r2_te,
        "mu": mu, "sd": sd, "history": history
    }

# =========================
# MAIN: un solo experimento para data.csv
# =========================
def main():
    CSV_PATH = "data.csv"   # tu nuevo dataset
    TARGET = "Grades"
    FEATURES_LIST = ["Socioeconomic Score", "Study Hours", "Sleep Hours", "Attendance (%)"]

    _ = run_experiment(
        CSV_PATH, FEATURES_LIST, TARGET,
        alpha=5e-2, epochs=4000, tol=1e-10, verbose=False, plot=True,
        title="Regresión Lineal - data.csv", hist_mode="raw"  # cambia a "std" si prefieres
    )

if __name__ == "__main__":
    main()
