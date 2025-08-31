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
    

# Funcion de Costo MSE
def mse_cost_from_preds(preds, Y):
    m = len(Y)

    # Errores (h - y)
    errors = [preds[i] - Y[i] for i in range(m)]

    # Errores² y suma 
    sq_errors = [e**2 for e in errors]
    total = sum(sq_errors)

    # Costo final
    cost = total / (2*m) # J = (1/(2m)) Σ (h - y)^2

    return cost, errors


def grad_Descent(X, Y, theta, b, alpha):

    # Derivada respecto a bias
    preds = hyp(X, theta, b)
    cost, errors = mse_cost_from_preds(preds, Y)

    m,n = len(Y), len(theta)
    db = sum(errors) / m
    print("Derivada Respecto a Bias:", db)

    # Recorrer cada parametro de theta
    dtheta = []
    for j in range(n):
        s = 0.0
        for i in range(m):
            s += errors[i] * X[i][j]
        s /= m
        dtheta.append(s)
    print("Derivada respecto a Theta:", dtheta)

    theta_new = [theta[j] - alpha * dtheta[j] for j in range(n)]
    b_new = b - alpha * db

     # (Opcional) costo después de actualizar
    preds_new = hyp(X, theta_new, b_new)
    cost_new, _ = mse_cost_from_preds(preds_new, Y)
    print(f"Costo antes: {cost:.6f} | Costo después: {cost_new:.6f}")

    return theta_new, b_new, dtheta, db




def main():
    # === Datos ===
    # X = Caracteristicas de Entrada, valores que usamos para predecir
    X = [[59], [44], [51], [42]]   # características
    # Y = Valores a los cuales queremos aproximar
    Y = [60, 55, 50, 66]           # valores reales
    
    # Parámetros iniciales
    theta = [0.3]
    b = 2.0
    alpha = 0.00001


    # === Ejecución ===
    # Función de hipótesis
    result_hyp = hyp(X, theta, b)
    print("① Predicciones hθ(x):", result_hyp)

    # Función de Costo inicial
    cost, errors = mse_cost_from_preds(result_hyp, Y)
    print("② Errores (hθ(xᵢ) - yᵢ):", errors)
    sq_errors = [e**2 for e in errors]
    print("③ Errores²:", sq_errors)
    total = sum(sq_errors)
    print("④ Suma Σ errores²:", total)
    print(f"⑤ Costo J(θ,b) = 1/(2·{len(Y)}) * {total:.4f} = {cost:.4f}")
    print("Costo inicial (MSE):", cost)

    grad_Descent(X, Y, theta, b, alpha)



# Ejecutar solo si es main
if __name__ == "__main__":
    main()
