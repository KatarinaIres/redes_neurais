import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler
import time

# Carregar os dados
data = load_diabetes()
X = data.data
y = data.target

# Aplicar normalização Min-Max
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Adicionar coluna de 1s
X_b = np.c_[np.ones((X_normalized.shape[0], 1)), X_normalized]

# Equação normal
start_time = time.time()
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
end_time = time.time()

# Predição e erro
y_pred = X_b.dot(theta)
mse = np.mean((y - y_pred) ** 2)

# Resultados
print("Parâmetros (theta):", theta)
print("Tempo de execução:", round(end_time - start_time, 4), "s")
print("Erro quadrático médio (MSE):", round(mse, 4))
