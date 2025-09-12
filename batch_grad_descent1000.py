import csv, math, os, contextlib

# =========================
# Funciones base (hipótesis, coste, gradiente)
# =========================
def hyp(x, theta, b):
    if isinstance(x[0], list):
        results = []
        for sample in x:
            y_hat = sum(sample[i] * theta[i] for i in range(len(sample))) + b
            results.append(y_hat)
        return results
    else:
        return sum(x[i] * theta[i] for i in range(len(x))) + b

def mse_cost_from_preds(preds, Y):
    m = len(Y)
    errors = [preds[i] - Y[i] for i in range(m)]
    sq_errors = [e**2 for e in errors]
    total = sum(sq_errors)
    cost = total / (2*m)
    return cost, errors

def grad_Descent(X, Y, theta, b, alpha):
    preds = hyp(X, theta, b)
    cost, errors = mse_cost_from_preds(preds, Y)

    m,n = len(Y), len(theta)
    db = sum(errors) / m

    dtheta = []
    for j in range(n):
        s = sum(errors[i] * X[i][j] for i in range(m)) / m
        dtheta.append(s)

    theta_new = [theta[j] - alpha * dtheta[j] for j in range(n)]
    b_new = b - alpha * db

    preds_new = hyp(X, theta_new, b_new)
    cost_new, _ = mse_cost_from_preds(preds_new, Y)
    return theta_new, b_new, dtheta, db

# =========================
# Configuración para data.csv
# =========================
FEATURES = ["Socioeconomic Score", "Study Hours", "Sleep Hours", "Attendance (%)"]
TARGET   = "Grades"

def load_dataset(path="data.csv"):
    X, Y = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = [float(row[k]) for k in FEATURES]
            y = float(row[TARGET])
            X.append(x)
            Y.append(y)
    return X, Y

# =========================
# Funciones auxiliares
# =========================
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
    alpha=1e-4,
    num_iters=200_000,
    log_each=10_000
):
    n = len(X[0])
    if theta is None:
        theta = [0.0] * n

    history = []
    for it in range(1, num_iters + 1):
        # Ejecutar un paso de gradiente
        theta, b, dtheta, db = grad_Descent(X, Y, theta, b, alpha)

        # Logging ocasional
        if it == 1 or it % log_each == 0 or it == num_iters:
            J = compute_cost(X, Y, theta, b)
            history.append(J)
            print(f"iter {it:>7} | J={J:.6f} | b={b:.6f} | theta={theta}")

    return theta, b, history

def predict_score(socio, study, sleep, attend, theta, b):
    x = [socio, study, sleep, attend]
    return hyp(x, theta, b)

# =========================
# Ejecución
# =========================
def run_training(csv_path="data.csv", alpha=1e-4, iters=200_000, log_each=20_000):
    X, Y = load_dataset(csv_path)

    theta0 = [0.0] * len(FEATURES)
    b0 = 0.0

    print(f"Filas: {len(Y)} | Features: {len(FEATURES)} -> {FEATURES}")
    print("Entrenando (Batch GD, sin estandarizar)...")
    theta, b, history = train_linear_regression_batch(
        X, Y, theta=theta0, b=b0, alpha=alpha,
        num_iters=iters, log_each=log_each
    )

    metrics, _ = evaluate_model(X, Y, theta, b)
    print("\n=== RESULTADOS ===")
    print(f"b (bias): {b:.6f}")
    print("theta:", [round(t, 6) for t in theta])
    print("Métricas:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()})

    return theta, b, metrics, history

# Ejecutar entrenamiento
theta, b, metrics, history = run_training("data.csv", alpha=1e-4, iters=200_000, log_each=20_000)
print(theta, b, metrics)
