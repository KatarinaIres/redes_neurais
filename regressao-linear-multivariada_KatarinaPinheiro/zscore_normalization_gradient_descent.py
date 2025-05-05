import numpy as np
import time
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# Dados
data = load_diabetes()
X = data.data
y = data.target

# Normalização Z-score
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Adiciona a coluna de 1s (termo de bias)
X_b = np.c_[np.ones((X_normalized.shape[0], 1)), X_normalized]

# Parâmetros
n_iterations = 1000
learning_rate = 0.01
m = X_b.shape[0]
theta = np.random.randn(X_b.shape[1])

# Gradiente Descendente
start_time = time.time()
for iteration in range(n_iterations):
    gradients = (2/m) * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients
end_time = time.time()

# Previsão e erro
y_pred = X_b.dot(theta)
mse = np.mean((y - y_pred) ** 2)

# Resultados
print("Parâmetros (theta):", theta)
print("Tempo de execução:", round(end_time - start_time, 4), "s")
print("Erro quadrático médio (MSE):", round(mse, 4))