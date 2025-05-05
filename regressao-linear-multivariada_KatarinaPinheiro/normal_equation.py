import numpy as np
from sklearn.datasets import load_diabetes
import time

# Carregar os dados
data = load_diabetes()
X = data.data
y = data.target

# Adicionar coluna de 1s para o termo de bias (intercepto)
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Aplicar a equação normal
start_time = time.time()
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
end_time = time.time()

# Predição e cálculo do erro
y_pred = X_b.dot(theta)
mse = np.mean((y - y_pred) ** 2)

# Resultados
print("Parâmetros (theta):", theta)
print("Tempo de execução:", round(end_time - start_time, 4), "s")
print("Erro quadrático médio (MSE):", round(mse, 4))
