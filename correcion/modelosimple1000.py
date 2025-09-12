# ============================
#  Regresión lineal (Batch GD)
#  Dataset: data.csv
#  X: Socioeconomic Score, Study Hours, Sleep Hours, Attendance (%)
#  Y: Grades
#  (sin estandarización)
# ============================

import csv, math, os, contextlib

# ---------- Núcleo que ya tenías ----------
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
    

# Función de costo MSE (formulación J = (1/2m) * Σ (h - y)^2)
def mse_cost_from_preds(preds, Y):
    m = len(Y)
    errors = [preds[i] - Y[i] for i in range(m)]
    sq_errors = [e**2 for e in errors]
    total = sum(sq_errors)
    cost = total / (2*m)
    return cost, errors


def grad_Descent(X, Y, theta, b, alpha):
    # Derivadas de una iteración de Batch GD
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

    # (Opcional) costo después de actualizar (silenciado por el runner)
    return theta_new, b_new, dtheta, db

# ---------- Adaptación al nuevo dataset ----------

# Los nombres exactos de las columnas tal como aparecen en el CSV
FEATURES = ["Socioeconomic Score", "Study Hours", "Sleep Hours", "Attendance (%)"]
TARGET   = "Grades"

# Alias/variantes por si el CSV tiene mayúsculas distintas o espacios variables
HEADER_ALIASES = {
    "socioeconomic score": "Socioeconomic Score",
    "study hours": "Study Hours",
    "sleep hours": "Sleep Hours",
    "attendance (%)": "Attendance (%)",
    "attendance %": "Attendance (%)",
    "grades": "Grades"
}

def _normalize_header(h):
    return h.strip().lower().replace("\ufeff", "")

def _remap_headers(fieldnames):
    """
    Convierte fieldnames arbitrarios a los esperados usando HEADER_ALIASES.
    Devuelve un dict: {nombre_original_csv -> nombre_estándar}
    """
    mapping = {}
    for raw in fieldnames:
        key = _normalize_header(raw)
        std = HEADER_ALIASES.get(key, None)
        if std is None:
            # Si no está en aliases, devuelve el original tal cual
            std = raw
        mapping[raw] = std
    return mapping

def load_data(path="data.csv"):
    """
    Carga data.csv y construye X (lista de listas) y Y (lista).
    Intenta mapear encabezados a los esperados.
    """
    X, Y = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Remapeo de encabezados a nombres estándar
        header_map = _remap_headers(reader.fieldnames)

        for row in reader:
            # Reconstruir la fila con headers estándar
            std_row = { header_map[k]: v for k, v in row.items() }

            # Extraer features en el orden definido por FEATURES
            x = []
            for k in FEATURES:
                try:
                    x.append(float(str(std_row[k]).strip()))
                except (KeyError, ValueError, TypeError):
                    raise ValueError(f"No pude leer la columna '{k}'. Revisa encabezados y datos numéricos.")
            try:
                y = float(str(std_row[TARGET]).strip())
            except (KeyError, ValueError, TypeError):
                raise ValueError(f"No pude leer la columna target '{TARGET}'.")
            X.append(x)
            Y.append(y)
    return X, Y


def compute_cost(X, Y, theta, b):
    preds = hyp(X, theta, b)
    cost, _ = mse_cost_from_preds(preds, Y)
    return cost


def evaluate_model(X, Y, theta, b):
    preds = hyp(X, theta, b)
    m = len(Y)
    J, errs = mse_cost_from_preds(preds, Y)
    mse  = sum(e*e for e in errs) / m
    rmse = math.sqrt(mse)
    mae  = sum(abs(e) for e in errs) / m
    ybar = sum(Y) / m
    ss_tot = sum((y - ybar)**2 for y in Y)
    ss_res = sum(e*e for e in errs)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")
    return {"J": J, "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}, preds


def train_linear_regression_batch(
    X, Y,
    theta=None, b=0.0,
    alpha=1e-4,        # LR pequeño porque no hay estandarización
    num_iters=200_000,
    log_each=10_000,
    mute_inner=True
):
    n = len(X[0])
    if theta is None:
        theta = [0.0] * n

    history = []
    for it in range(1, num_iters + 1):
        if mute_inner:
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                theta, b, dtheta, db = grad_Descent(X, Y, theta, b, alpha)
        else:
            theta, b, dtheta, db = grad_Descent(X, Y, theta, b, alpha)

        if it == 1 or it % log_each == 0 or it == num_iters:
            J = compute_cost(X, Y, theta, b)
            history.append(J)
            print(f"iter {it:>7} | J={J:.6f} | b={b:.6f} | theta={theta}")

    return theta, b, history


def predict_grade(
    socioeconomic_score, study_hours, sleep_hours, attendance_percent,
    theta, b
):
    x = [socioeconomic_score, study_hours, sleep_hours, attendance_percent]
    return hyp(x, theta, b)

# ---------- Runner de ejemplo ----------

def run_training(
    csv_path="data.csv",
    alpha=1e-4,
    iters=200_000,
    log_each=20_000
):
    X, Y = load_data(csv_path)

    theta0 = [0.0] * len(FEATURES)
    b0 = 0.0

    print(f"Filas: {len(Y)} | Features: {len(FEATURES)} -> {FEATURES}")
    print("Entrenando (Batch GD, sin estandarizar)...")

    theta, b, history = train_linear_regression_batch(
        X, Y, theta=theta0, b=b0,
        alpha=alpha, num_iters=iters,
        log_each=log_each, mute_inner=True
    )

    metrics, _ = evaluate_model(X, Y, theta, b)
    print("\n=== RESULTADOS ===")
    print(f"b (bias): {b:.6f}")
    print("theta:", [round(t, 6) for t in theta])
    print("Métricas:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()})

    return theta, b, metrics, history

# Ejecutar entrenamiento si corres este archivo directamente:
if __name__ == "__main__":
    theta, b, metrics, history = run_training("data.csv", alpha=1e-4, iters=200_000, log_each=20_000)
    print(theta, b, metrics)
