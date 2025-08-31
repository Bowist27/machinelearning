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
    

def mse_cost(X, Y, theta, b):
    m = len(Y)
    preds = hyp(X, theta, b)

    # 1) Predicciones
    print("① Predicciones hθ(x):", preds)

    # 2) Errores (h - y)
    errors = [preds[i] - Y[i] for i in range(m)]
    print("② Errores (hθ(xᵢ) - yᵢ):", errors)

    # 3) Errores al cuadrado
    sq_errors = [e**2 for e in errors]
    print("③ Errores²:", sq_errors)

    # 4) Suma de errores²
    total = sum(sq_errors)
    print("④ Suma Σ errores²:", total)

    # 5) Costo final
    cost = total / (2*m)
    print(f"⑤ Costo J(θ,b) = 1/(2·{m}) * {total:.4f} = {cost:.4f}")

    return cost

